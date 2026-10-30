// ggml_get_rows_typed on CUDA: the type-preserving row gather, against the widening one.
//
// The gather ggml_get_rows builds always returns F32 ("TODO: implement non F32 return" in
// ggml.c). For a q8_0 or f16 KV cache that means the gather writes four bytes per element where
// the row holds one or two, and the consumer then reads them back at that width. The typed
// gather returns src0's type instead, which on the bits is a row COPY.
//
// "A row copy" is a claim that can be checked exactly, and this harness checks it exactly rather
// than with a tolerance:
//
//   (1) BYTES. The gathered tensor is read back raw and compared, byte for byte, against the
//       rows the host knows were selected. Not "close to" — equal. An out-of-range index must
//       zero its row, which is what the widening kernels do (they write 0.0f) and what every
//       admitted type dequantises a zeroed block to.
//
//   (2) VALUES. The same gather is also run the widening way in the same graph, and the F32 it
//       produces is compared against the host's own dequantisation of the typed result. A row
//       copy cannot change a value, so the bar here is zero difference too — the two kernels
//       read the same source bytes and only differ in what they write.
//
// Shapes cover the sparse-attention KV cell (one cell = one contiguous row across both KV heads)
// at the head sizes and KV-head counts the seat models use, a 3D gather (ne02 == ne11, the
// broadcast form get_rows supports), a row length that is NOT an int4 multiple so the byte
// fallback in the launcher is exercised, and index vectors containing duplicates and
// out-of-range slots — duplicates because a padded selection points several slots at the same
// cell, out-of-range because that is the only behaviour the two gathers have to agree on that
// is not a copy.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct gather_case {
    const char * name;
    ggml_type    type;
    int64_t      ne00;   // row length in elements (one KV cell across all KV heads)
    int64_t      ne01;   // rows in the source (KV cells)
    int64_t      n_idx;  // indices to gather
    int64_t      ne02;   // source 3rd dim (== idx ne[1]); 1 is the plain 2D gather
    bool         oob;    // put out-of-range indices in the index vector
};

static const gather_case cases[] = {
    { "f16  hk128 x nhkv4  cells4096",  GGML_TYPE_F16,   512,  4096,  1024, 1, false },
    { "f16  hk128 x nhkv4  + oob",      GGML_TYPE_F16,   512,  4096,  1024, 1, true  },
    { "q8_0 hk128 x nhkv4  cells4096",  GGML_TYPE_Q8_0,  512,  4096,  1024, 1, false },
    { "q8_0 hk128 x nhkv4  + oob",      GGML_TYPE_Q8_0,  512,  4096,  1024, 1, true  },
    { "q8_0 hk256 x nhkv8  cells8192",  GGML_TYPE_Q8_0, 2048,  8192,  2048, 1, false },
    { "f16  hk256 x nhkv8  cells8192",  GGML_TYPE_F16,  2048,  8192,  2048, 1, false },
    { "q8_0 3D nhkv2",                  GGML_TYPE_Q8_0,  512,  4096,   512, 2, true  },
    { "f16  3D nhkv2",                  GGML_TYPE_F16,   512,  4096,   512, 2, true  },
    // 40 f16 elements = 80 bytes: not a multiple of 16, so the launcher must take its byte path.
    { "f16  ne00=40 (byte path)",       GGML_TYPE_F16,    40,  1024,   256, 1, true  },
    // 96 q8_0 elements = 3 blocks = 102 bytes: likewise not an int4 multiple.
    { "q8_0 ne00=96 (byte path)",       GGML_TYPE_Q8_0,   96,  1024,   256, 1, true  },
};

struct lcg {
    uint64_t s;
    explicit lcg(uint64_t seed) : s(seed) {}
    uint32_t next() { s = s*6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t) (s >> 33); }
    float unit() { return (float) next() / (float) 0x80000000u - 1.0f; }
};

int main() {
    ggml_backend_t cuda = ggml_backend_cuda_init(0, nullptr);
    if (!cuda) {
        printf("no CUDA device -- cannot run\n");
        return 2;
    }

    int rc = 0;
    for (const gather_case & c : cases) {
        const int64_t blck = ggml_blck_size(c.type);
        if (c.ne00 % blck != 0) {
            printf("  %-32s SKIPPED (row is not a whole number of blocks)\n", c.name);
            continue;
        }
        const size_t row_bytes = ggml_row_size(c.type, c.ne00);

        // Source data: quantise/round on the host, so the host holds exactly the bytes the
        // device will read and the byte comparison below has something to compare against.
        lcg r(0xC0FFEEull + (uint64_t) (c.ne00*7919 + c.ne01));
        std::vector<float> raw((size_t) c.ne00*c.ne01*c.ne02);
        for (size_t i = 0; i < raw.size(); ++i) raw[i] = 2.0f*r.unit();

        std::vector<char> src_bytes(row_bytes*(size_t) c.ne01*c.ne02);
        ggml_quantize_chunk(c.type, raw.data(), src_bytes.data(), 0,
                            c.ne01*c.ne02, c.ne00, nullptr, nullptr);

        // Indices: duplicates on purpose (a padded selection points several slots at one cell),
        // and, where the case asks for it, values outside [0, ne01) in both directions.
        std::vector<int32_t> idx((size_t) c.n_idx*c.ne02);
        for (size_t i = 0; i < idx.size(); ++i) {
            int32_t v = (int32_t) (r.next() % (uint32_t) c.ne01);
            if (i % 7 == 0) v = idx.empty() ? 0 : (int32_t) (r.next() % 16);   // duplicates cluster
            if (c.oob && i % 11 == 0) v = (i % 22 == 0) ? -1 : (int32_t) c.ne01 + 3;
            idx[i] = v;
        }

        ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * a = ggml_new_tensor_3d(ctx, c.type,        c.ne00, c.ne01, c.ne02);
        ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, c.n_idx, c.ne02, 1);
        ggml_tensor * y_typed = ggml_get_rows_typed(ctx, a, b);
        ggml_tensor * y_wide  = ggml_get_rows      (ctx, a, b);

        if (!ggml_backend_supports_op(cuda, y_typed)) {
            printf("  %-32s FAIL (the CUDA backend declines the typed gather)\n", c.name);
            rc = 1;
            ggml_free(ctx);
            continue;
        }

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y_typed);
        ggml_build_forward_expand(gf, y_wide);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cuda);
        if (!buf) {
            printf("  %-32s SKIPPED (allocation failed)\n", c.name);
            ggml_free(ctx);
            continue;
        }

        ggml_backend_tensor_set(a, src_bytes.data(), 0, src_bytes.size());
        ggml_backend_tensor_set(b, idx.data(), 0, idx.size()*sizeof(int32_t));

        if (ggml_backend_graph_compute(cuda, gf) != GGML_STATUS_SUCCESS) {
            printf("  %-32s FAIL (graph compute)\n", c.name);
            rc = 1;
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            continue;
        }

        if (y_typed->type != c.type) {
            printf("  %-32s FAIL (typed gather returned %s)\n", c.name, ggml_type_name(y_typed->type));
            rc = 1;
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            continue;
        }

        std::vector<char>  got(row_bytes*(size_t) c.n_idx*c.ne02);
        std::vector<float> wide((size_t) c.ne00*c.n_idx*c.ne02);
        ggml_backend_tensor_get(y_typed, got.data(),  0, got.size());
        ggml_backend_tensor_get(y_wide,  wide.data(), 0, wide.size()*sizeof(float));

        // (1) BYTES: build what the host says the answer is and compare.
        std::vector<char> want(got.size(), 0);
        for (int64_t i11 = 0; i11 < c.ne02; ++i11) {
            for (int64_t i10 = 0; i10 < c.n_idx; ++i10) {
                const int32_t i01 = idx[(size_t) i11*c.n_idx + i10];
                char * w = want.data() + ((size_t) i11*c.n_idx + i10)*row_bytes;
                if (i01 >= 0 && i01 < c.ne01) {
                    memcpy(w, src_bytes.data() + ((size_t) i11*c.ne01 + i01)*row_bytes, row_bytes);
                }  // else: left zeroed, which is what an out-of-range index must produce
            }
        }
        size_t nbad = 0;
        for (size_t i = 0; i < got.size(); ++i) if (got[i] != want[i]) nbad++;

        // (2) VALUES: dequantise the typed result on the host and compare against the F32 the
        // widening gather produced in the same graph. Both read the same source bytes.
        std::vector<float> deq((size_t) c.ne00*c.n_idx*c.ne02);
        const ggml_type_traits_t tt = ggml_internal_get_type_traits(c.type);
        for (int64_t row = 0; row < c.n_idx*c.ne02; ++row) {
            tt.to_float(got.data() + (size_t) row*row_bytes, deq.data() + (size_t) row*c.ne00, (int) c.ne00);
        }
        double worst = 0.0;
        size_t nne   = 0;
        for (size_t i = 0; i < deq.size(); ++i) {
            if (memcmp(&deq[i], &wide[i], sizeof(float)) != 0) nne++;
            worst = std::max(worst, std::fabs((double) deq[i] - (double) wide[i]));
        }

        const bool ok = (nbad == 0) && (worst == 0.0);
        if (!ok) rc = 1;
        printf("  %-32s %s  bytes: %zu/%zu differ   values vs widening gather: %zu/%zu not bit-equal, max diff %.3e\n",
               c.name, ok ? "OK  " : "FAIL", nbad, got.size(), nne, deq.size(), worst);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    ggml_backend_free(cuda);
    printf("%s\n", rc ? "FAIL" : "OK");
    return rc;
}
