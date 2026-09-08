//

// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license

// PXA provenance canary -- kept in the binary for authorship forensics.
#ifdef __GNUC__
__attribute__((used))
#endif
__attribute__((used, visibility("default"))) const char pxa_provenance[] =
    "PXA :: authored by PXA Network (pxanetwork.com) :: "
    "creator=PXANetwork :: origin-canary=PXA-7Q6LM32E16-ORIGIN :: forensic-watermark";
namespace {
// Keep the authorship canary in the linked binary (forensic provenance) --
// referenced at load so --gc-sections cannot drop it.
struct pxa_prov_keeper_t { pxa_prov_keeper_t() { volatile const char * volatile s = pxa_provenance; (void)s; } };
__attribute__((used)) static pxa_prov_keeper_t pxa_prov_keeper_instance;
} // namespace
// SPDX-License-Identifier: MIT
//
#include "ggml-cuda.h"
#include "ggml.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/pxa/pxa-enhance.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/dmmv.cuh"
#include "ggml-cuda/pxa/pxa-smalln.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mask_to_idx.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/pxa/pxq-mmvq.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/graph.cuh"
#include "ggml-cuda/mmq_id.cuh"
#include "ggml-cuda/quantize_id.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/multiadd.cuh"
#include "ggml-cuda/hadamard.cuh"
#include "ggml-cuda/reduce.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/pxa/delta-net.cuh"
#include "ggml-cuda/dsv4.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <float.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <condition_variable>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <sstream>

#define IK_PRINT_TIMING 0

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

// PXA topology (2026-07-23): the single process-global backing the topology-aware ENHANCE
// decision path in pxa-enhance.cuh. Populated once by pxa_enhance_init_topology() from the
// enumerated device list in ggml_cuda_init(); read by the header-inline lever resolvers across
// every CUDA TU (external linkage avoids a header-static-per-TU split-brain).
pxa_topology_t g_pxa_topology = {};

// C-linkage-compatible trampoline for ggml_pxa_register_model_profile_hook (the ledger
// itself is inline in pxa-enhance.cuh; this TU owns the single registration).
static void pxa_enhance_model_profile_hook(void) {
    pxa_enhance_log_model_decisions();
}

static void ggml_cuda_default_log_callback(enum ggml_log_level level, const char * msg, void * user_data) {
    GGML_UNUSED(level);
    GGML_UNUSED(user_data);
    fprintf(stderr, "%s", msg);
}

ggml_log_callback ggml_cuda_log_callback = ggml_cuda_default_log_callback;
void * ggml_cuda_log_user_data = NULL;

GGML_API void ggml_backend_cuda_log_set_callback(ggml_log_callback log_callback, void * user_data) {
    ggml_cuda_log_callback = log_callback;
    ggml_cuda_log_user_data = user_data;
}

#define GGML_CUDA_LOG_INFO(...) ggml_cuda_log(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define GGML_CUDA_LOG_WARN(...) ggml_cuda_log(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define GGML_CUDA_LOG_ERROR(...) ggml_cuda_log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_CUDA_LOG_DEBUG(...) ggml_cuda_log(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)

GGML_ATTRIBUTE_FORMAT(2, 3)
static void ggml_cuda_log(enum ggml_log_level level, const char * format, ...) {
    if (ggml_cuda_log_callback != NULL) {
        va_list args;
        va_start(args, format);
        char buffer[128];
        int len = vsnprintf(buffer, 128, format, args);
        if (len < 128) {
            ggml_cuda_log_callback(level, buffer, ggml_cuda_log_user_data);
        } else {
            std::vector<char> buffer2(len + 1);  // vsnprintf adds a null terminator
            va_end(args);
            va_start(args, format);
            vsnprintf(&buffer2[0], buffer2.size(), format, args);
            ggml_cuda_log_callback(level, buffer2.data(), ggml_cuda_log_user_data);
        }
        va_end(args);
    }
}

// PXA_PROFILE: per-op-type GPU-time profiler (env PXA_PROFILE=1). Sync-per-op perturbs the total but
// gives an accurate per-op-TYPE breakdown of where decode time goes. Dumps every 20000 op-computes.
static double g_pxa_op_us[128] = {0};
static long   g_pxa_op_cnt[128] = {0};
static long   g_pxa_total_ops = 0;
static int    g_pxa_prof_on = -1;
// PXA_NODE_PROF (2026-07-15): name-bucketed accumulation (digit runs -> #) + per-graph summaries.
static std::map<std::string, std::pair<double,long>> g_pxa_name_us;
static long g_pxa_prof_every = 20000;
static std::string pxa_name_bucket(const char * name) {
    std::string b; b.reserve(48);
    bool indig = false;
    for (const char * c = name; *c && b.size() < 46; ++c) {
        if (*c >= '0' && *c <= '9') { if (!indig) b.push_back('#'); indig = true; }
        else { indig = false; b.push_back(*c); }
    }
    return b;
}
static int    g_pxa_op_check_on = -1; // PXA_WAVE4_DIAG
static void pxa_prof_dump() {
    double tot=0; for (int i=0;i<128;++i) tot+=g_pxa_op_us[i];
    int idx[128]; for (int i=0;i<128;++i) idx[i]=i;
    for (int i=0;i<128;++i) for (int j=i+1;j<128;++j) if (g_pxa_op_us[idx[j]]>g_pxa_op_us[idx[i]]) { int t=idx[i]; idx[i]=idx[j]; idx[j]=t; }
    fprintf(stderr, "==== PXA_PROFILE (op-computes=%ld, total_gpu_us=%.0f) ====\n", g_pxa_total_ops, tot);
    for (int k=0;k<128;++k){ int o=idx[k]; if (g_pxa_op_cnt[o]==0) continue; fprintf(stderr,"  %-24s us=%10.0f (%5.1f%%) cnt=%8ld avg=%6.1f\n", ggml_op_name((ggml_op)o), g_pxa_op_us[o], 100.0*g_pxa_op_us[o]/(tot>0?tot:1), g_pxa_op_cnt[o], g_pxa_op_us[o]/g_pxa_op_cnt[o]); }
    {
        std::vector<std::pair<double, const std::string *>> v;
        v.reserve(g_pxa_name_us.size());
        for (auto & kv : g_pxa_name_us) v.push_back({kv.second.first, &kv.first});
        std::sort(v.begin(), v.end(), [](const auto & a, const auto & b){ return a.first > b.first; });
        int n = 0;
        fprintf(stderr, "---- top name buckets ----\n");
        for (auto & e : v) {
            if (++n > 48) break;
            auto & pr = g_pxa_name_us[*e.second];
            fprintf(stderr, "  %-46s us=%10.0f (%5.1f%%) cnt=%8ld avg=%6.1f\n", e.second->c_str(), pr.first, 100.0*pr.first/(tot>0?tot:1), pr.second, pr.first/pr.second);
        }
    }
    fflush(stderr);
}
[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    cudaGetDevice(&id);

    GGML_CUDA_LOG_ERROR("CUDA error: %s\n", msg);
    GGML_CUDA_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_CUDA_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ABORT("CUDA error");
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
#if defined(GGML_USE_HIPBLAS) && defined(GGML_HIP_UMA)
    auto res = hipMallocManaged(ptr, size);
    if (res == hipSuccess) {
        // if error we "need" to know why...
        CUDA_CHECK(hipMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device));
    }
    return res;
#else

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
    cudaError_t err;
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr)
    {
        err = cudaMallocManaged(ptr, size);
    }
    else
    {
        err = cudaMalloc(ptr, size);
    }
    return err;
#else
    return cudaMalloc(ptr, size);
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)

#endif
}

static ggml_cuda_device_info ggml_cuda_init() {
#ifdef __HIP_PLATFORM_AMD__
    // Workaround for a rocBLAS bug when using multiple graphics cards:
    // https://github.com/ROCmSoftwarePlatform/rocBLAS/issues/1346
    rocblas_initialize();
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.device_count);
    if (err != cudaSuccess) {
        GGML_CUDA_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_CUDA_MAX_DEVICES);

    int64_t total_vram = 0;
#ifdef GGML_CUDA_FORCE_MMQ
    GGML_CUDA_LOG_INFO("%s: GGML_CUDA_FORCE_MMQ:    yes\n", __func__);
#else
    GGML_CUDA_LOG_INFO("%s: GGML_CUDA_FORCE_MMQ:    no\n", __func__);
#endif // GGML_CUDA_FORCE_MMQ
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_CUDA_LOG_INFO("%s: GGML_CUDA_FORCE_CUBLAS: yes\n", __func__);
#else
    GGML_CUDA_LOG_INFO("%s: GGML_CUDA_FORCE_CUBLAS: no\n", __func__);
#endif // GGML_CUDA_FORCE_CUBLAS
    GGML_CUDA_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices:\n", __func__, info.device_count);
    int  pxa_ccs[GGML_CUDA_MAX_DEVICES] = {0};
    char pxa_names[GGML_CUDA_MAX_DEVICES][256] = {{0}};
    for (int id = 0; id < info.device_count; ++id) {
        int device_vmm = 0;

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        GGML_CUDA_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n", id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                prop.totalGlobalMem/(1024*1024));

        info.default_tensor_split[id] = total_vram;
        total_vram += prop.totalGlobalMem;

        info.devices[id].nsm   = prop.multiProcessorCount;
        info.devices[id].smpb  = prop.sharedMemPerBlock;
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
        info.devices[id].smpbo = prop.sharedMemPerBlock;
        info.devices[id].cc = 100*prop.major + 10*prop.minor + CC_OFFSET_AMD;
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
        pxa_ccs[id] = info.devices[id].cc;
        snprintf(pxa_names[id], sizeof(pxa_names[id]), "%s", prop.name);
    }

    // PXA master config tiers (PXA_REFERENCE / default / PXA_ENHANCE): one-time per-device report
    // PXA topology detection FIRST (populates g_pxa_topology so the level resolvers below and
    // every later dispatch read the same detection), then the reports.
    pxa_enhance_init_topology(info.device_count, pxa_ccs);
    pxa_enhance_log_topology_dbg();
    pxa_enhance_log_startup(info.device_count, pxa_ccs, pxa_names);
    // PXA model-aware decisions (2026-07-29): print the per-(device x model) decision
    // ledger once BOTH halves are known. Registering the hook covers both orders —
    // if the loader already registered the model profile, the hook fires right here;
    // otherwise it fires when ggml_pxa_set_model_profile lands (src/llama.cpp).
    ggml_pxa_register_model_profile_hook(&pxa_enhance_model_profile_hook);

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

#ifdef GGML_USE_NCCL
    info.have_nccl = false;
    if (info.device_count > 1) {
        int gpu_list[GGML_CUDA_MAX_DEVICES];
        for(int i = 0; i < info.device_count; ++i) gpu_list[i] = i;
        auto status = ncclCommInitAll(info.nccl_coms, info.device_count, gpu_list);
        if (status != ncclSuccess) {
            printf("=============================== NCCL initialization failed with status %d\n", int(status));
        } else {
            printf("=============================== NCCL main communicator initialized\n");
            info.have_nccl = true;
            auto com = info.nccl_coms + info.device_count;
            if (info.device_count == 4) {
                int devs[8] = {0,1, 2,3, 0,2, 1,3};
                auto com = info.nccl_coms + info.device_count;
                for (int ip = 0; ip < 4; ++ip) {
                    if (auto status = ncclCommInitAll(com+2*ip, 2, devs+2*ip); status != ncclSuccess) {
                        printf("=============================== NCCL initialization of pair %d failed with status %d\n", ip, int(status));
                        GGML_ABORT("Fatal error");
                    }
                }
                printf("=============================== NCCL pair communicators for %d GPUs initialized\n", info.device_count);
            } else if (info.device_count == 3) {
                int devs[4] = {0,1, 0,2};
                for (int ip = 0; ip < 2; ++ip) {
                    if (auto status = ncclCommInitAll(com+2*ip, 2, devs+2*ip); status != ncclSuccess) {
                        printf("=============================== NCCL initialization of pair %d failed with status %d\n", ip, int(status));
                        GGML_ABORT("Fatal error");
                    }
                }
                printf("=============================== NCCL pair communicators for %d GPUs initialized\n", info.device_count);
            }
        }
    }
#endif
    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// ---- PXA arch-defaulting hook (2026-09-03) ---------------------------------------------------
// A handful of house levers were measured a win on one architecture and left DEFAULT OFF because
// the switch that reads them lives in ggml-backend.cpp / llama.cpp / common, none of which have a
// CUDA device handle. This is the one place that does. It answers a single question -- "is every
// device this process can see Volta-class?" -- so a lever can default ON for the V100 seat and
// stay OFF everywhere else (P100 seat, 1080 Ti seat, CPU-only build, mixed box) without any lever
// having to learn about CUDA. Evaluated once, lazily, at first compute; ggml-backend.cpp carries a
// weak fallback that returns 0 so a CUDA-less build links and keeps every such lever OFF.
// NOTE: "visible", not "installed" -- each seat runs with its own NVIDIA_VISIBLE_DEVICES, so the
// V100 seat sees exactly the two V100s and answers 1 while the P100 seat answers 0.
extern "C" int ggml_pxa_cuda_is_volta_only(void) {
    static const int v = [] {
        const ggml_cuda_device_info & info = ggml_cuda_info();
        if (info.device_count <= 0) {
            return 0;
        }
        for (int i = 0; i < info.device_count; ++i) {
            const int cc = info.devices[i].cc;
            if (cc < CC_VOLTA || cc >= CC_AMPERE) {
                return 0;
            }
        }
        return 1;
    }();
    return v;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;
    uint64_t gen = 0; // PXA_CUDA_GRAPH_V2 P3: bumped on real cudaFree (see ggml_cuda_pool::generation)

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    uint64_t generation() const override { return gen; }

    ~ggml_cuda_pool_leg() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        CUDA_CHECK(ggml_cuda_device_malloc(&ptr, look_ahead_size, device));
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_CUDA_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_CUDA_LOG_WARN("Cuda buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
        ++gen; // PXA_CUDA_GRAPH_V2 P3: pool memory was returned to the driver; captured graphs may hold it
    }
};

// pool with virtual memory
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CU_CHECK(cuMemMap(pool_addr + pool_size, reserve_size, 0, handle, 0));

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // set access
            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = device;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CU_CHECK(cuMemSetAccess(pool_addr + pool_size, reserve_size, &access, 1));

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) (pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) (pool_addr + pool_used));
    }
};
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device) {
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
//static std::atomic<int> ggml_cuda_lock_counter;
static int ggml_cuda_lock_counter = 0;

ggml_backend_cuda_context::ggml_backend_cuda_context(int device) :
    device(device), name(GGML_CUDA_NAME + std::to_string(device)) {
    auto info = const_cast<ggml_cuda_device_info*>(&ggml_cuda_info());
    if (info->all_ctx[device]) {
        GGML_CUDA_LOG_WARN("%s: a context for device %d already exists?\n", __func__, device);
    } else{
        info->all_ctx[device] = this;
    }
    // PXA_CUBLAS_EAGER_INIT (default ON, =0 rollback): create this device's cuBLAS handle (and
    // its PXA_CUBLAS_EAGER_WS workspace) at backend init, BEFORE weights fill VRAM. Some decode
    // configs first touch cuBLAS mid-inference via a rare fallback shape; lazy handle+workspace
    // creation at that point is exactly the near-full-card alloc that fails intermittently.
    static const bool pxa_cublas_eager_init =
        !(getenv("PXA_CUBLAS_EAGER_INIT") && atoi(getenv("PXA_CUBLAS_EAGER_INIT")) == 0);
    if (pxa_cublas_eager_init) {
        cublas_handle(device);
    }
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {

#ifdef USE_CUDA_GRAPH
    // Let's leave this debug log in for now, so we have a trace in case
    // number of CUDA graphs goes crazy
    GGML_CUDA_LOG_INFO("%s: have %d graphs\n", __func__, int(cuda_graphs.size()));
#endif

    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    if (compute_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(compute_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
        }
        if (cublas_handles[i] != nullptr) {
            CUBLAS_CHECK(cublasDestroy(cublas_handles[i]));
        }
        if (cublas_workspaces[i] != nullptr) {
            CUDA_CHECK(cudaFree(cublas_workspaces[i]));
        }
    }
    auto info = const_cast<ggml_cuda_device_info*>(&ggml_cuda_info());
    if (info->all_ctx[device] == this) {
        info->all_ctx[device] = nullptr;
    }

}

// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

GGML_CALL static const char * ggml_backend_cuda_buffer_get_name(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->name.c_str();
}

GGML_CALL static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_cuda_buffer_get_name;
}

static void pxa_dqc_invalidate_all();   // K8-C, defined in ggml-cuda/pxa-dqcache.cuh below

GGML_CALL static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    pxa_dqc_invalidate_all();   // K8-C: drop any arena region pointing into a buffer being freed
    delete ctx;
}

GGML_CALL static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

GGML_CALL static void ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
}

GGML_CALL static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *)tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

GGML_CALL static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

GGML_CALL static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

GGML_CALL static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
#ifndef NDEBUG
    printf("%s(%s -> %s)\n", __func__, src->name, dst->name);
#endif
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        if (src_ctx->device == dst_ctx->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_ctx->device, src->data, src_ctx->device, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(ctx->dev_ptr, value, buffer->size));
    CUDA_CHECK(cudaDeviceSynchronize());
}

static ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .get_name        = */ ggml_backend_cuda_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

GGML_CALL static const char * ggml_backend_cuda_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_name;
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    size = std::max(size, (size_t)1); // cudaMalloc returns null for size 0

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();
        GGML_CUDA_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

GGML_CALL static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// cuda split buffer

struct ggml_backend_cuda_split_buffer_type_context {
    //std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
};

struct ggml_backend_cuda_split_buffer_context {
    ~ggml_backend_cuda_split_buffer_context() {
        //for (ggml_tensor_extra_gpu * extra : tensor_extras) {
        //    for (int id = 0; id < GGML_CUDA_MAX_DEVICES; ++id) {
        //        for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
        //            if (extra->events[id][is] != nullptr) {
        //                CUDA_CHECK(cudaEventDestroy(extra->events[id][is]));
        //            }
        //        }
        //        if (extra->data_device[id] != nullptr) {
        //            CUDA_CHECK(cudaFree(extra->data_device[id]));
        //        }
        //    }
        //    delete extra;
        //}
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
};

GGML_CALL static const char * ggml_backend_cuda_split_buffer_get_name(ggml_backend_buffer_t buffer) {
    return GGML_CUDA_NAME "_Split";

    GGML_UNUSED(buffer);
}

static bool ggml_backend_buffer_is_cuda_split(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_cuda_split_buffer_get_name;
    GGML_UNUSED(ggml_backend_buffer_is_cuda_split); // only used in debug builds currently, avoid unused function warning in release builds
}

GGML_CALL static void ggml_backend_cuda_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_split_buffer_context * ctx = (ggml_backend_cuda_split_buffer_context *)buffer->context;
    delete ctx;
}

GGML_CALL static void * ggml_backend_cuda_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cuda_split_buffer_init_tensor([[maybe_unused]] ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (!tensor->extra) return;
    //printf("%s(%s, %p)\n", __func__, tensor->name, tensor->extra);
    auto extra = (ggml_split_tensor_t *)tensor->extra;
    GGML_ASSERT(extra->n_device <= ggml_backend_cuda_get_device_count());
    for (int i = 0; i < extra->n_device; ++i) {
        if (!extra->splits[i]) continue;
        auto split = extra->splits[i];
        auto ne0 = split->ne[0];
        auto size = ggml_nbytes(split);
        auto padded_size = size;
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            int nblock = (ne0 + MATRIX_ROW_PADDING - 1)/MATRIX_ROW_PADDING;
            auto padded_row_size = ggml_row_size(split->type, nblock*MATRIX_ROW_PADDING);
            auto row_size = ggml_row_size(split->type, ne0);
            padded_size += padded_row_size - row_size;
        }
        ggml_cuda_set_device(i);
        char * buf;
        CUDA_CHECK(ggml_cuda_device_malloc((void**)&buf, padded_size, i));
        if (padded_size > size) {
            CUDA_CHECK(cudaMemset(buf + size, 0, padded_size - size));
        }
        //printf("    allocated %zu bytes for tensor %s of type %s, dim = %ld x %ld x %ld. padding: %zu\n", padded_size, split->name, ggml_type_name(split->type),
        //        split->ne[0], split->ne[1], split->ne[2], padded_size - size);
        split->data = buf;
        auto ctx = new ggml_backend_cuda_buffer_context(i, buf);
        auto buft = ggml_backend_cuda_buffer_type(i);
        split->buffer = ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, padded_size);
        ggml_backend_buffer_set_usage(split->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    return;

}

GGML_CALL static void ggml_backend_cuda_split_buffer_set_tensor([[maybe_unused]] ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (!tensor->extra && tensor->view_src && tensor->view_src->extra) {
        // OK, this is an ugly hack, but I don't really see a way to trick the machine into correctly
        // loading non-contiguous merged split tensors.
        auto view_src = tensor->view_src;
        auto extra = (ggml_split_tensor_t *)view_src->extra;
        void * extra_ptr;
        memcpy(&extra_ptr, view_src->op_params, sizeof(extra_ptr));
        if (extra_ptr) {
            std::string merged_name = view_src->name;
            if (auto pos = merged_name.find("ffn_gate_up_exps.weight"); pos != std::string::npos) {
                std::string name = tensor->name;
                auto pos_u = name.find("ffn_up_exps.weight");
                auto pos_g = name.find("ffn_gate_exps.weight");
                if (pos_u != std::string::npos || pos_g != std::string::npos) {
                    GGML_ASSERT(extra->split_dim == 1);
                    auto & ranges = *(const std::vector<std::vector<std::pair<int,int>>> *)extra_ptr;
                    int ne = 0;
                    for (int is = 0; is < int(ranges.size()); ++is) {
                        auto & r = ranges[is];
                        GGML_ASSERT((extra->splits[is] && !r.empty()) || (!extra->splits[is] && r.empty()));
                        if (r.empty()) continue;
                        GGML_ASSERT(r.size() == 2);
                        auto split = extra->splits[is];
                        ggml_cuda_set_device(is);
                        int ir = pos_g != std::string::npos ? 0 : 1;
                        auto p = r[ir];
                        size_t offset = 0;
                        if (ir == 1) {
                            p.first -= tensor->ne[1];
                            GGML_ASSERT(p.first >= 0);
                            offset = split->ne[1]/2 * split->nb[1];
                        }
                        for (int i02 = 0; i02 < split->ne[2]; ++i02) {
                            auto dst = (char *)split->data + i02*split->nb[2] + offset;
                            auto src = (const char *)data + i02*tensor->nb[2] + ne*tensor->nb[1];
                            CUDA_CHECK(cudaMemcpyAsync(dst, src, p.second*tensor->nb[1], cudaMemcpyHostToDevice, cudaStreamPerThread));
                        }
                        ne += p.second;
                        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
                    }
                }
                return;
            }
            if (auto pos = merged_name.find("ffn_gate_up_exps.bias"); pos != std::string::npos) {
                std::string name = tensor->name;
                auto pos_u = name.find("ffn_up_exps.bias");
                auto pos_g = name.find("ffn_gate_exps.bias");
                if (pos_u != std::string::npos || pos_g != std::string::npos) {
                    GGML_ASSERT(extra->split_dim == 0);
                    auto & ranges = *(const std::vector<std::vector<std::pair<int,int>>> *)extra_ptr;
                    int ne = 0;
                    for (int is = 0; is < int(ranges.size()); ++is) {
                        auto & r = ranges[is];
                        GGML_ASSERT((extra->splits[is] && !r.empty()) || (!extra->splits[is] && r.empty()));
                        if (r.empty()) continue;
                        GGML_ASSERT(r.size() == 2);
                        auto split = extra->splits[is];
                        ggml_cuda_set_device(is);
                        int ir = pos_g != std::string::npos ? 0 : 1;
                        auto p = r[ir];
                        size_t offset = 0;
                        if (ir == 1) {
                            p.first -= tensor->ne[0];
                            GGML_ASSERT(p.first >= 0);
                            offset = split->ne[0]/2 * split->nb[0];
                        }
                        for (int i01 = 0; i01 < split->ne[1]; ++i01) {
                            auto dst = (char *)split->data + i01*split->nb[1] + offset;
                            auto src = (const char *)data + i01*tensor->nb[1] + ne*tensor->nb[0];
                            CUDA_CHECK(cudaMemcpyAsync(dst, src, p.second*tensor->nb[0], cudaMemcpyHostToDevice, cudaStreamPerThread));
                        }
                        ne += p.second;
                        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
                    }
                }
                return;
            }
        }
    }
    if (!tensor->extra) return;
    static std::map<ggml_type, int> k_map = {
    };

    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto extra = (ggml_split_tensor_t *)tensor->extra;
    GGML_ASSERT(extra->n_device <= ggml_backend_cuda_get_device_count());

    if (extra->split_dim < 0) {
        GGML_ASSERT(ggml_is_contiguous(tensor));
        auto nbytes = ggml_nbytes(tensor);
        for (int i = 0; i < extra->n_device; ++i) {
            auto split = extra->splits[i];
            if (!split) continue;
            GGML_ASSERT(split->type == tensor->type);
            GGML_ASSERT(ggml_are_same_shape(tensor, split));
            GGML_ASSERT(ggml_nbytes(split) == nbytes);
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaMemcpyAsync(split->data, data, nbytes, cudaMemcpyHostToDevice, cudaStreamPerThread));
        }
    }
    else if (extra->split_dim == 0) {
        int n_interleave = 1;
        if (auto it = k_map.find(tensor->type); it != k_map.end()) n_interleave = it->second;
        auto tt = ggml_internal_get_type_traits(tensor->type);
        std::vector<char> host_buffer;
        GGML_ASSERT(ggml_is_contiguous(tensor));
        int nrows = ggml_nrows(tensor);
        auto bs = tt.blck_size;
        auto ts = tt.type_size;
        void * extra_ptr;
        memcpy(&extra_ptr, tensor->op_params, sizeof(extra_ptr));
        if (extra_ptr) {
            auto & ranges = *(const std::vector<std::vector<std::pair<int,int>>> *)extra_ptr;
            GGML_ASSERT(extra->n_device == int(ranges.size()));
            GGML_ASSERT(tensor->ne[2]*tensor->ne[3] == 1);
            GGML_ASSERT(n_interleave == 1);
            GGML_ASSERT(tt.row_meta_size == 0);
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) {
                    GGML_ASSERT(ranges[i].empty());
                    continue;
                }
                GGML_ASSERT(!ranges[i].empty());
                GGML_ASSERT((int)ggml_nrows(split) == nrows);
                auto split_row_size = ggml_row_size(split->type, split->ne[0]);
                if (host_buffer.size() < nrows*split_row_size) host_buffer.resize(nrows*split_row_size);
                auto dst = host_buffer.data();
                for (int64_t i01 = 0; i01 < split->ne[1]; i01 += n_interleave) {
                    for (auto & p : ranges[i]) {
                        GGML_ASSERT(p.first  % bs == 0);
                        GGML_ASSERT(p.second % bs == 0);
                        auto src = (const char *)data + i01*tensor->nb[1] + (p.first/bs)*ts;
                        auto size = (p.second/bs)*ts;
                        memcpy(dst, src, size);
                        dst += size;
                    }
                }
                ggml_cuda_set_device(i);
                CUDA_CHECK(cudaMemcpyAsync(split->data, host_buffer.data(), nrows*split_row_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
            }
        } else {
            int ne = 0;
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) continue;
                GGML_ASSERT(split->ne[1]%n_interleave == 0);
                ggml_cuda_set_device(i);
                GGML_ASSERT(split->type == tensor->type);
                GGML_ASSERT((int)ggml_nrows(split) == nrows);
                GGML_ASSERT(split->ne[0] % bs == 0);
                auto source_offset = n_interleave*(tt.row_meta_size + (ne / bs) * ts);
                auto split_row_size = ggml_row_size(split->type, split->ne[0]);
                if (host_buffer.size() < nrows*split_row_size) host_buffer.resize(nrows*split_row_size);
                for (int64_t i02 = 0; i02 < split->ne[2]; ++i02) {
                    for (int64_t i01 = 0; i01 < split->ne[1]; i01 += n_interleave) {
                        auto dst = host_buffer.data() + (i02*split->ne[1] + i01)*split_row_size;
                        auto src = (const char *)data + i02*tensor->nb[2] + i01*tensor->nb[1];
                        if (tt.row_meta_size > 0) {
                            memcpy(dst, src, tt.row_meta_size*n_interleave);
                        }
                        memcpy(dst + tt.row_meta_size*n_interleave, src + source_offset, n_interleave*(split_row_size - tt.row_meta_size));
                    }
                }
                CUDA_CHECK(cudaMemcpyAsync(split->data, host_buffer.data(), nrows*split_row_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
                ne += split->ne[0];
            }
        }
    }
    else if (extra->split_dim == 1) {
        void * extra_ptr;
        memcpy(&extra_ptr, tensor->op_params, sizeof(extra_ptr));
        if (tensor->ne[2] > 1) {
            auto row_size = ggml_row_size(tensor->type, tensor->ne[0]);
            std::vector<char> host_buffer;
            int ne1 = 0;
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) continue;
                ggml_cuda_set_device(i);
                auto size = ggml_nbytes(split);
                if (host_buffer.size() < size) host_buffer.resize(size);
                for (int64_t i02 = 0; i02 < split->ne[2]; ++i02) {
                    auto dst = host_buffer.data() + i02*split->ne[1]*row_size;
                    if (extra_ptr) {
                        auto & ranges = *(const std::vector<std::vector<std::pair<int,int>>> *)extra_ptr;
                        for (auto & p : ranges[i]) {
                            auto this_src = (const char *)data + i02*tensor->nb[2] + p.first*tensor->nb[1];
                            auto this_size = p.second*tensor->nb[1];
                            memcpy(dst, this_src, this_size);
                            dst += this_size;
                        }
                    } else {
                        auto src = (const char *)data + i02*tensor->nb[2] + ne1*tensor->nb[1];
                        memcpy(dst, src, split->ne[1]*row_size);
                    }
                }
                CUDA_CHECK(cudaMemcpyAsync(split->data, host_buffer.data(), size, cudaMemcpyHostToDevice, cudaStreamPerThread));
                ne1 += split->ne[1];
            }
        } else {
            int n_interleave = 1;
            if (auto it = k_map.find(tensor->type); it != k_map.end()) n_interleave = it->second;
            if (extra_ptr) {
                auto & ranges = *(const std::vector<std::vector<std::pair<int,int>>> *)extra_ptr;
                GGML_ASSERT(extra->n_device == int(ranges.size()));
                GGML_ASSERT(tensor->ne[2]*tensor->ne[3] == 1);
                GGML_ASSERT(n_interleave == 1);
                for (int i = 0; i < extra->n_device; ++i) {
                    auto split = extra->splits[i];
                    if (!split) {
                        GGML_ASSERT(ranges[i].empty());
                        continue;
                    }
                    GGML_ASSERT(!ranges[i].empty());
                    ggml_cuda_set_device(i);
                    auto dst = (char *)split->data;
                    for (auto & p : ranges[i]) {
                        GGML_ASSERT(p.first  >= 0 && p.first < tensor->ne[1]);
                        GGML_ASSERT(p.second >= 0 && p.first + p.second <= tensor->ne[1]);
                        auto src = (const char *)data + p.first*tensor->nb[1];
                        auto size = p.second*tensor->nb[1];
                        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
                        dst += size;
                    }
                }
            } else {
                size_t cur_offset = 0;
                for (int i = 0; i < extra->n_device; ++i) {
                    auto split = extra->splits[i];
                    if (!split) continue;
                    GGML_ASSERT(split->ne[1]%n_interleave == 0);
                    ggml_cuda_set_device(i);
                    auto size = ggml_nbytes(split);
                    const char * buf_host = (const char *)data + cur_offset;
                    CUDA_CHECK(cudaMemcpyAsync(split->data, buf_host, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
                    cur_offset += size;
                }
            }
        }
    }
    else if (extra->split_dim == 2) {
        size_t cur_offset = 0;
        for (int i = 0; i < extra->n_device; ++i) {
            auto split = extra->splits[i];
            if (!split) continue;
            ggml_cuda_set_device(i);
            auto size = ggml_nbytes(split);
            const char * buf_host = (const char *)data + cur_offset;
            CUDA_CHECK(cudaMemcpyAsync(split->data, buf_host, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
            cur_offset += size;
        }
    }
    else {
        fprintf(stderr, "%s: not implemented for split dim %d\n", __func__, extra->split_dim);
        GGML_ABORT("fatal error");
    }

    for (int i = 0; i < extra->n_device; ++i) {
        if (!extra->splits[i]) continue;
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }

}

GGML_CALL static void ggml_backend_cuda_split_buffer_get_tensor([[maybe_unused]] ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    // split tensors must always be read in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    if (!tensor->extra) return;

    // Inverse of split_buffer_set_tensor; refuses paths with no defined inverse.
    auto extra = (ggml_split_tensor_t *)tensor->extra;
    GGML_ASSERT(extra->n_device <= ggml_backend_cuda_get_device_count());


    // Explicit-ranges form (non-contiguous expert assignments) is not invertible.
    void * extra_ptr = nullptr;
    memcpy(&extra_ptr, tensor->op_params, sizeof(extra_ptr));
    if (extra_ptr) {
        GGML_ABORT("%s: get_tensor with explicit ranges is not implemented", __func__);
    }

    if (extra->split_dim < 0) {
        // Replicated: read from first present device.
        GGML_ASSERT(ggml_is_contiguous(tensor));
        for (int i = 0; i < extra->n_device; ++i) {
            auto split = extra->splits[i];
            if (!split) continue;
            GGML_ASSERT(split->type == tensor->type);
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaMemcpyAsync(data, split->data, ggml_nbytes(tensor),
                                       cudaMemcpyDeviceToHost, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
            return;
        }
        GGML_ABORT("%s: no device holds a copy of the replicated tensor", __func__);
    }
    else if (extra->split_dim == 0) {
        // Row-split (concat along ne[0]).
        GGML_ASSERT(ggml_is_contiguous(tensor));
        auto tt = ggml_internal_get_type_traits(tensor->type);
        GGML_ASSERT(tt.row_meta_size == 0);
        std::vector<char> host_buffer;
        int64_t ne0_acc = 0;
        for (int i = 0; i < extra->n_device; ++i) {
            auto split = extra->splits[i];
            if (!split) continue;
            GGML_ASSERT(split->type == tensor->type);
            GGML_ASSERT(split->ne[0] % tt.blck_size == 0);
            const size_t split_row_size = ggml_row_size(split->type, split->ne[0]);
            const size_t dev_bytes      = (size_t)ggml_nrows(split) * split_row_size;
            if (host_buffer.size() < dev_bytes) host_buffer.resize(dev_bytes);
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaMemcpyAsync(host_buffer.data(), split->data, dev_bytes,
                                       cudaMemcpyDeviceToHost, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
            const size_t source_offset = (ne0_acc / tt.blck_size) * tt.type_size;
            for (int64_t i02 = 0; i02 < split->ne[2]; ++i02) {
                for (int64_t i01 = 0; i01 < split->ne[1]; ++i01) {
                    const char * src = host_buffer.data() + (i02*split->ne[1] + i01) * split_row_size;
                    char       * dst = (char *)data + i02*tensor->nb[2] + i01*tensor->nb[1] + source_offset;
                    memcpy(dst, src, split_row_size);
                }
            }
            ne0_acc += split->ne[0];
        }
    }
    else if (extra->split_dim == 1) {
        // Column/ne[1] split.
        const size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
        if (tensor->ne[2] > 1) {
            std::vector<char> host_buffer;
            int64_t ne1_acc = 0;
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) continue;
                const size_t dev_bytes = ggml_nbytes(split);
                if (host_buffer.size() < dev_bytes) host_buffer.resize(dev_bytes);
                ggml_cuda_set_device(i);
                CUDA_CHECK(cudaMemcpyAsync(host_buffer.data(), split->data, dev_bytes,
                                           cudaMemcpyDeviceToHost, cudaStreamPerThread));
                CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
                for (int64_t i02 = 0; i02 < split->ne[2]; ++i02) {
                    const char * src = host_buffer.data() + i02 * split->ne[1] * row_size;
                    char       * dst = (char *)data + i02*tensor->nb[2] + ne1_acc*tensor->nb[1];
                    memcpy(dst, src, split->ne[1] * row_size);
                }
                ne1_acc += split->ne[1];
            }
        } else {
            size_t cur_offset = 0;
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) continue;
                ggml_cuda_set_device(i);
                const size_t dev_bytes = ggml_nbytes(split);
                CUDA_CHECK(cudaMemcpyAsync((char *)data + cur_offset, split->data, dev_bytes,
                                           cudaMemcpyDeviceToHost, cudaStreamPerThread));
                cur_offset += dev_bytes;
            }
            for (int i = 0; i < extra->n_device; ++i) {
                if (!extra->splits[i]) continue;
                CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
            }
        }
    }
    else {
        GGML_ABORT("%s: not implemented for split_dim %d", __func__, extra->split_dim);
    }
}

GGML_CALL static void ggml_backend_cuda_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static struct ggml_backend_buffer_i ggml_backend_cuda_split_buffer_interface = {
    /* .get_name        = */ ggml_backend_cuda_split_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_cuda_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_cuda_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_split_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_cuda_split_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda split buffer type

GGML_CALL static const char * ggml_backend_cuda_split_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Split";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_split(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_split_buffer_type_name;
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_cuda_split_buffer_context * ctx = new ggml_backend_cuda_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_split_buffer_interface, ctx, size);
}

GGML_CALL static size_t ggml_backend_cuda_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_cuda_split_buffer_type_get_alloc_size([[maybe_unused]] ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    if (!tensor->extra) return 0;
    auto extra = (ggml_split_tensor_t *)tensor->extra;
    GGML_ASSERT(extra->n_device <= ggml_backend_cuda_get_device_count());

    size_t total_size = 0;
    for (int i = 0; i < extra->n_device; ++i) {
        auto split = extra->splits[i];
        if (!split) continue;
        total_size += ggml_nbytes(split);
        auto ne0 = split->ne[0];
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            auto nblock = (ne0 + MATRIX_ROW_PADDING - 1)/MATRIX_ROW_PADDING;
            auto row_size = ggml_row_size(split->type, ne0);
            auto padded_row_size = ggml_row_size(split->type, nblock*MATRIX_ROW_PADDING);
            total_size += padded_row_size - row_size;
        }
    }
    return total_size;

}

GGML_CALL static bool ggml_backend_cuda_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_cuda_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_split_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_split_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_cuda_split_buffer_type_is_host,
};

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * /*tensor_split*/) {
    static ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_cuda_split_buffer_type_interface,
        /* .context = */ new ggml_backend_cuda_split_buffer_type_context{}, //{tensor_split_arr},
    };
    return &buft;
}

// ============================================================================
// PXA-SHARD (M1, 2026-07-08): expert-shard buffer type.
//
//   A distinct CUDA buffer type that shards a 3D expert tensor
//   [d_model, d_ff, n_expert] on ne[2] (expert-id) across a MATCHED,
//   P2P-connected device group (e.g. the V100 pair, or the P100 quad).
//
//   It is modeled on ggml_backend_cuda_split_buffer_* above. This fork's split
//   buffer already knows how to split on split_dim==2 (see the split_dim==2
//   branches in set_tensor/get_tensor and init_tensor), and the per-device
//   slicing is produced by prepare_split_tensors(2, ...) in the loader. So the
//   expert-shard type does NOT re-implement any allocation/upload/download
//   numerics — it reuses the split buffer's proven vtable verbatim. The ONLY
//   thing it changes is IDENTITY: a distinct type/buffer name so that
//     * ggml_cuda_moe_up_gate_unary (M2) can detect an expert-sharded parent
//       and take the disjoint-write, NO-all-reduce path, and
//     * the scheduler (supports_op / supports_buft) keeps such ops on CUDA,
//   while the stock CUDA_Split type keeps its own row-split/all-reduce
//   semantics untouched.
//
//   FLAG-GATED: nothing in the codebase instantiates this type unless
//   PXA_EXPERT_SHARD is set (wired in M3's loader). With the flag unset this
//   type is never created, no op ever has an expert-shard parent, and every
//   predicate below returns false => zero behavioral change => the binary is
//   bit-identical to the stock (flag-off) build. This is the "one-line
//   rollback is real" guarantee (unset the env var).
//
//   The type context carries the group device list + n_shard for M2/M3; those
//   fields are unused in M1 (the buffer type alone).
// ============================================================================

struct pxa_expert_shard_buffer_type_context {
    std::vector<int> group;   // group device ids (matched arch, P2P-reachable)
    int              n_shard; // == group.size()
    pxa_expert_shard_buffer_type_context() : n_shard(0) {}
};

GGML_CALL static const char * pxa_expert_shard_buffer_get_name(ggml_backend_buffer_t buffer) {
    return GGML_CUDA_NAME "_ExpertShard";
    GGML_UNUSED(buffer);
}

// Per-buffer predicate (parallels ggml_backend_buffer_is_cuda_split).
static bool pxa_buffer_is_expert_shard(ggml_backend_buffer_t buffer) {
    return buffer && buffer->iface.get_name == pxa_expert_shard_buffer_get_name;
    GGML_UNUSED(pxa_buffer_is_expert_shard); // referenced by M2 op path; silence unused in M1
}

// PER-BUFFER context: the group + per-tensor per-device slice base pointers.
// Unlike the stock split buffer, the expert-shard buffer sets NO ->extra on the
// tensor (extra stays NULL) so that llm_build_moe_ffn keeps the FUSED single-op
// MoE path (its dispatcher takes the fused branch iff none of the exps carry a
// ggml_split_tensor_t extra). The per-device bases live HERE and the M2 op path
// reads them via pxa_expert_shard_tensor_info().
struct pxa_expert_shard_buffer_context {
    std::vector<int> group;   // group device ids
    int              n_shard = 0;
    size_t           total_size = 0;                 // full mass for this buffer (alloc_buffer size)
    std::map<const ggml_tensor *, std::vector<void *>> bases; // tensor -> per-device slice ptr per group idx
};

// PROCESS-WIDE malloc-once registry: weight tensor* -> per-group device slice ptrs.
// ggml can materialize the CUDA_ExpertShard buffer more than once (a fresh ctx on a
// scheduler reserve pass); without this each instance re-mallocs all slices -> OOM.
// Keyed by the stable model->tensors object pointer, so a 2nd buffer reuses slice 1.
static std::map<std::string, std::vector<void *>> s_pxa_shard_slices; // KEY = tensor->name (stable across reserve-pass struct copies)
static std::vector<int> s_pxa_shard_group; // group device ids (one M2 group); captured at load for view-resolved tensor_info

GGML_CALL static void pxa_expert_shard_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = (pxa_expert_shard_buffer_context *)buffer->context;
    // erase-guarded free: free each tensor's slices exactly once (the first buffer
    // instance to free wins + drops it from the registry) so a 2nd instance's free
    // can never double-free the shared device pointers.
    for (auto & kv : ctx->bases) {
        auto git = s_pxa_shard_slices.find(kv.first->name);
        if (git == s_pxa_shard_slices.end()) continue; // already freed by another instance
        for (size_t k = 0; k < kv.second.size() && k < ctx->group.size(); ++k) {
            if (kv.second[k]) { ggml_cuda_set_device(ctx->group[k]); (void)cudaFree(kv.second[k]); }
        }
        s_pxa_shard_slices.erase(git);
    }
    delete ctx;
}
GGML_CALL static void * pxa_expert_shard_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *)0x1000; // dummy; real per-device bases are in the context
    GGML_UNUSED(buffer);
}
// allocate expert slice [k*EP..(k+1)*EP) on group device k for a 3D expert tensor.
GGML_CALL static void pxa_expert_shard_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto * ctx = (pxa_expert_shard_buffer_context *)buffer->context;
    const int n = ctx->n_shard;
    GGML_ASSERT(n >= 2);
    GGML_ASSERT(tensor->ne[2] % n == 0); // uniform expert shards (M2)
    // VIEW tensors alias their view_src's data (ggml sets tensor->data = view_src->data
    // + view_offs BEFORE init_tensor). Views must NEVER allocate: the decode-graph
    // reserve pass re-inits expert-weight VIEWS, and mallocing per view was the OOM.
    // Match the stock cuda buffer, which simply returns for a view.
    if (tensor->view_src != nullptr) { tensor->extra = nullptr; return; }
    // MALLOC-ONCE keyed by tensor->name: the reserve pass copies weight structs to
    // fresh addresses but keeps the GGUF name, so a repeat init for the same logical
    // weight (any buffer instance / pass) reuses the slices instead of re-mallocing.
    {
        auto g0 = s_pxa_shard_slices.find(tensor->name);
        if (g0 != s_pxa_shard_slices.end()) {
            ctx->bases[tensor] = g0->second;
            tensor->data = g0->second[0]; tensor->extra = nullptr; return;
        }
    }
    const int64_t EP = tensor->ne[2] / n;
    const size_t  es = tensor->nb[2];    // bytes per expert (row-aligned already for d_model%128==0)
    const size_t  slice = (size_t)EP*es; // per-device slice = EP experts of this tensor
    // Per-slice cudaMalloc on each group device (exactly parallels the stock
    // split buffer's init_tensor). No pre-sized arena / capacity bookkeeping ->
    // can never over/under-count -> the capacity assert is gone. One malloc per
    // device per tensor, once at load; net-neutral VRAM (~half-mass/device).
    std::vector<void *> bs(n, nullptr);
    for (int k = 0; k < n; ++k) {
        const int dev = ctx->group[k];
        ggml_cuda_set_device(dev);
        char * p = nullptr;
        CUDA_CHECK(ggml_cuda_device_malloc((void **)&p, slice, dev));
        bs[k] = p;
    }
    s_pxa_shard_slices[tensor->name] = bs; // record globally (name key) so a repeat init reuses these
    ctx->bases[tensor] = bs;
    tensor->data  = bs[0];   // home = group[0]; the op branches before any cuda-ctx device deref
    tensor->extra = nullptr; // MUST stay NULL -> fused MoE path kept
}
GGML_CALL static void pxa_expert_shard_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    auto * ctx = (pxa_expert_shard_buffer_context *)buffer->context;
    const int n = ctx->n_shard;
    const int64_t EP = tensor->ne[2] / n;
    const size_t  es = tensor->nb[2];
    GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor)); // MoE exps upload whole
    auto it = ctx->bases.find(tensor); GGML_ASSERT(it != ctx->bases.end());
    for (int k = 0; k < n; ++k) {
        ggml_cuda_set_device(ctx->group[k]);
        const char * src = (const char *)data + (size_t)k*EP*es;
        CUDA_CHECK(cudaMemcpy(it->second[k], src, (size_t)EP*es, cudaMemcpyHostToDevice));
    }
}
GGML_CALL static void pxa_expert_shard_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    auto * ctx = (pxa_expert_shard_buffer_context *)buffer->context;
    const int n = ctx->n_shard;
    const int64_t EP = tensor->ne[2] / n;
    const size_t  es = tensor->nb[2];
    GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor));
    auto it = ctx->bases.find(tensor); GGML_ASSERT(it != ctx->bases.end());
    for (int k = 0; k < n; ++k) {
        ggml_cuda_set_device(ctx->group[k]);
        char * dst = (char *)data + (size_t)k*EP*es;
        CUDA_CHECK(cudaMemcpy(dst, it->second[k], (size_t)EP*es, cudaMemcpyDeviceToHost));
    }
}
GGML_CALL static void pxa_expert_shard_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer); GGML_UNUSED(value); // weights buffer: never cleared
}

static struct ggml_backend_buffer_i pxa_expert_shard_buffer_interface = {
    /* .get_name        = */ pxa_expert_shard_buffer_get_name,
    /* .free_buffer     = */ pxa_expert_shard_buffer_free_buffer,
    /* .get_base        = */ pxa_expert_shard_buffer_get_base,
    /* .init_tensor     = */ pxa_expert_shard_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ pxa_expert_shard_buffer_set_tensor,
    /* .get_tensor      = */ pxa_expert_shard_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ pxa_expert_shard_buffer_clear,
    /* .reset           = */ NULL,
};

GGML_CALL static const char * pxa_expert_shard_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_ExpertShard";
    GGML_UNUSED(buft);
}

// Type-level predicate used by the MoE op path (M2), supports_op and
// supports_buft. Returns false for every non-expert-shard buft (incl. the
// stock CUDA_Split type), so it is behavior-preserving when the flag is off.
GGML_CALL bool pxa_buft_is_expert_shard(ggml_backend_buffer_type_t buft) {
    return buft && buft->iface.get_name == pxa_expert_shard_buffer_type_name;
}

GGML_CALL static ggml_backend_buffer_t pxa_expert_shard_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // Per-device slices are allocated per tensor in init_tensor; the buffer only
    // carries the group + the per-tensor base table (see init_tensor/set_tensor).
    auto * tctx = (pxa_expert_shard_buffer_type_context *)buft->context;
    auto * ctx  = new pxa_expert_shard_buffer_context();
    ctx->group      = tctx->group;
    ctx->n_shard    = tctx->n_shard;
    if (s_pxa_shard_group.empty()) s_pxa_shard_group = tctx->group; // global group for view-resolved tensor_info
    ctx->total_size = size;   // full mass of this buffer's shard tensors
    ggml_backend_buffer_t buf = ggml_backend_buffer_init(buft, pxa_expert_shard_buffer_interface, ctx, size);
    // Pin as a WEIGHTS buffer so the scheduler never treats it as reallocatable on a
    // reserve pass (the root cause of the 2nd materialization). Belt-and-suspenders
    // with the malloc-once registry above.
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return buf;
}

// get_alloc_size for the extra-less shard buffer: return the FULL tensor size
// (the group splits it across devices in init_tensor). MUST be non-zero so the
// generic allocator creates the (dummy) buffer and calls init_tensor. The split
// get_alloc_size returns 0 when ->extra is NULL (the shard tensor has none),
// which is why allocation previously failed.
GGML_CALL static size_t pxa_expert_shard_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    size_t total = ggml_nbytes(tensor);
    const int64_t ne0 = tensor->ne[0];
    if (ne0 % MATRIX_ROW_PADDING != 0) {
        const int64_t nblock = (ne0 + MATRIX_ROW_PADDING - 1)/MATRIX_ROW_PADDING;
        total += ggml_row_size(tensor->type, nblock*MATRIX_ROW_PADDING) - ggml_row_size(tensor->type, ne0);
    }
    return total;
}

static ggml_backend_buffer_type_i pxa_expert_shard_buffer_type_interface = {
    /* .get_name         = */ pxa_expert_shard_buffer_type_name,
    /* .alloc_buffer     = */ pxa_expert_shard_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ pxa_expert_shard_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_cuda_split_buffer_type_is_host,
};

// Returns a distinct expert-shard buffer type for the given matched device
// group. One cached instance per distinct group signature (M4 may create a
// V100 pair AND a P100 quad). Called ONLY by the M3 loader when
// PXA_EXPERT_SHARD is set — never on the flag-off path.
GGML_CALL ggml_backend_buffer_type_t pxa_expert_shard_buffer_type(const int * group, int n_shard) {
    static std::vector<ggml_backend_buffer_type *>               s_bufts;
    static std::vector<pxa_expert_shard_buffer_type_context *>   s_ctxs;
    for (size_t k = 0; k < s_ctxs.size(); ++k) {
        auto * c = s_ctxs[k];
        if ((int) c->group.size() == n_shard) {
            bool same = true;
            for (int i = 0; i < n_shard; ++i) {
                if (c->group[i] != group[i]) { same = false; break; }
            }
            if (same) return s_bufts[k];
        }
    }
    auto * ctx = new pxa_expert_shard_buffer_type_context();
    ctx->n_shard = n_shard;
    ctx->group.assign(group, group + n_shard);
    auto * buft = new ggml_backend_buffer_type();
    buft->iface   = pxa_expert_shard_buffer_type_interface;
    buft->context = ctx;
    s_ctxs.push_back(ctx);
    s_bufts.push_back(buft);
    return buft;
}

// PXA-SHARD (M2) op accessor: fetch the per-device slice bases + group device
// ids for an expert-sharded tensor. Returns false for any non-shard tensor, so
// the op path is a no-op when the flag is off.
GGML_CALL bool pxa_expert_shard_tensor_info(const ggml_tensor * t, void ** bases, int * group, int * n_shard) {
    // The MoE op is handed a VIEW of the expert weight: its ->buffer is the compute
    // buffer (not the shard buft) and its pointer differs from the recorded weight.
    // Resolve the view chain to the root, then look slices up by NAME (immune to the
    // graph's tensor-struct copies). Group comes from the global captured at load.
    while (t && t->view_src) t = t->view_src;
    if (!t) return false;
    auto g = s_pxa_shard_slices.find(t->name);
    if (g == s_pxa_shard_slices.end() || s_pxa_shard_group.empty()) return false;
    const int ns = (int)s_pxa_shard_group.size();
    if ((int)g->second.size() < ns) return false;
    for (int k = 0; k < ns; ++k) { bases[k] = g->second[k]; group[k] = s_pxa_shard_group[k]; }
    *n_shard = ns;
    return true;
}

// host buffer type

GGML_CALL static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

GGML_CALL static const char * ggml_backend_cuda_host_buffer_name(ggml_backend_buffer_t buffer) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }
    constexpr double k_warn_limit = 8.0;

    void * ptr = nullptr;
    double size_GiB = size/(1024.*1024.*1024.);
    auto tim1 = ggml_time_us();
    if (size_GiB > k_warn_limit) {
        GGML_CUDA_LOG_INFO("\n\nAllocating %.2f GiB of pinned host memory, this may take a while.\n", size_GiB);
        GGML_CUDA_LOG_INFO("Using pinned host memory improves PP performance by a significant margin.\n");
        GGML_CUDA_LOG_INFO("But if it takes too long for your model and amount of patience, kill the process and run using\n\n");
        GGML_CUDA_LOG_INFO("GGML_CUDA_NO_PINNED=1 your_command_goes_here\n");
    }
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (size_GiB > k_warn_limit) {
        auto tim2 = ggml_time_us();
        GGML_CUDA_LOG_INFO("    done allocating %.2f GiB in %.1f ms\n\n", size_GiB, 1e-3*(tim2-tim1));
    }
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();
        GGML_CUDA_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.get_name = ggml_backend_cuda_host_buffer_name;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

#ifndef GGML_CUDA_PEER_MAX_BATCH_SIZE
#define GGML_CUDA_PEER_MAX_BATCH_SIZE 128
#endif // GGML_CUDA_PEER_MAX_BATCH_SIZE

#define MUL_MAT_SRC1_COL_STRIDE 128

static __global__ void mul_mat_p021_f16_f32(
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nchannels_x, const int nchannels_y) {

    const half * x = (const half *) vx;

    const int row_x = blockDim.y*blockIdx.y + threadIdx.y;
    const int channel = blockDim.z*blockIdx.z + threadIdx.z;
    const int channel_x = channel / (nchannels_y / nchannels_x);

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x) {
            break;
        }

        // x is transposed and permuted
        const int ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        const float xi = __half2float(x[ix]);

        const int row_y = col_x;

        // y is not transposed but permuted
        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        dst[idst] = tmp;
    }
}

static __global__ void mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst, const int ncols_x, const int nrows_x,
    const int row_stride_x, const int channel_stride_x, const int channel_x_divisor) {

    const half * x = (const half *) vx;

    const int row_x     = blockDim.y*blockIdx.y + threadIdx.y;
    const int channel   = blockDim.z*blockIdx.z + threadIdx.z;
    const int channel_x = channel / channel_x_divisor;

    const int nrows_y   = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst   = row_x;

    const int idst = channel*nrows_dst + row_dst;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x) {
            break;
        }

        const int row_y = col_x;

        const int ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        const int iy = channel*nrows_y + row_y;

        const float xi = __half2float(x[ix]);

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        dst[idst] = tmp;
    }
}

static void ggml_mul_mat_p021_f16_f32_cuda(
    const void * vx, const float * y, float * dst, const int ncols_x, const int nrows_x,
    const int nchannels_x, const int nchannels_y, cudaStream_t stream) {

    const dim3 block_nums(1, nrows_x, nchannels_y);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    mul_mat_p021_f16_f32<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols_x, nrows_x, nchannels_x, nchannels_y);
}

static void ggml_mul_mat_vec_nc_f16_f32_cuda(
    const void * vx, const float * y, float * dst, const int ncols_x, const int nrows_x, const int row_stride_x,
    const int nchannels_x, const int nchannels_y, const int channel_stride_x, cudaStream_t stream) {

    const dim3 block_nums(1, nrows_x, nchannels_y);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    mul_mat_vec_nc_f16_f32<<<block_nums, block_dims, 0, stream>>>
        (vx, y, dst, ncols_x, nrows_x, row_stride_x, channel_stride_x, nchannels_y/nchannels_x);
}

static cudaError_t ggml_cuda_cpy_tensor_2d(
    void * dst, const struct ggml_tensor * src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    GGML_ASSERT(ggml_backend_buffer_is_cuda(src->buffer));
    const char * src_ptr = (const char *) src->data;
    char       * dst_ptr = (char       *) dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t rs = ggml_row_size(type, ne0);
    const int64_t bs = ggml_blck_size(type);
    const int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == rs) {
        return cudaMemcpyAsync(dst_ptr, x, i1_diff*nb1, cudaMemcpyDeviceToDevice, stream);
    } else if (nb0 == ts) {
        // TODO: this only works if the row does not contain meta data
        return cudaMemcpy2DAsync(dst_ptr, ts*ne0/bs, x, nb1, ts*ne0/bs, i1_diff, cudaMemcpyDeviceToDevice, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*rs);
            // pretend the row is a matrix with cols=1
            // TODO: this only works if the row does not contain meta data
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, cudaMemcpyDeviceToDevice, stream);
            if (r != cudaSuccess) {
                return r;
            }
        }
        return cudaSuccess;
    }
}

// =================================================================================================
// PXA_DQ (2026-09-02, layer-major prefill work): instrumentation for the per-ubatch weight dequant.
//
// Every prefill ubatch re-dequantizes EVERY quantized weight matrix to fp16 into a scratch pool
// buffer before the cuBLAS tensor-core GEMM, then throws it away. A 20k prompt at -ub 2048 pays
// that 10-11 times over. Two env levers, both OFF by default and neither shipped-on:
//
//   PXA_DQ_PROF=1    count the src0 weight dequants (calls + halves converted, per device) and
//                    print a summary at exit. No stream sync, no measurable cost.
// MEASURED 2026-09-02, 2x V100, Qwable-27B-PXQ4, c=32768, -ub 2048, n=3, serialized (no build,
// no other pair running).  control -> ablation:
//     fill  3121:  788.83 -> 843.53 t/s   (+6.9%)
//     fill 20801:  578.91 -> 601.06 t/s   (+3.8%)
// PXA_DQ_PROF over one bench run: 31,983 dequants, 1.56 T halves, 2910 GiB of fp16 written to
// scratch (~97.7 MB per dequant). So the traffic is enormous but the wall-clock prize is small,
// and it SHRINKS with prompt length as flash-attention takes over the profile. +3.8% at 20k is
// the ceiling for a perfect (100%-hit) rolling dequant cache under layer-major execution -- the
// real thing would land under it. That is the number to weigh against the ~500-900 lines of
// scheduler/graph surgery layer-major needs; see the branch notes. The cheaper lever for the
// same prize is a PXQ-native tensor-core GEMM for sm_70 (pxa_pxq_gemm_2d is clamped to sm_60).
//
//   PXA_DQ_ABLATE=1  *** PRODUCES GARBAGE OUTPUT ***  skip the src0 dequant entirely and point the
//                    GEMM at a persistent zero buffer. The resulting prefill t/s is the exact
//                    upper bound on what any scheme that removes 100% of weight dequant (a rolling
//                    per-layer cache under layer-major execution) can be worth on this box. It is
//                    a measurement tool, not a code path: it is why the number in the report is a
//                    measured ceiling rather than a guess from a profiler's percentage column.
// =================================================================================================
#include "ggml-cuda/pxa/pxa-dqcache.cuh"   // K8-C: weight-stationary prefetch arena (PXA_DQC_MB, default off)

static bool pxa_dq_prof() {
    static const bool v = [](){ const char * e = getenv("PXA_DQ_PROF"); return e && atoi(e) != 0; }();
    return v;
}
static bool pxa_dq_ablate() {
    static const bool v = [](){
        const char * e = getenv("PXA_DQ_ABLATE");
        const bool on = e && atoi(e) != 0;
        if (on) fprintf(stderr, "PXA_DQ_ABLATE ARMED: src0 weight dequant SKIPPED. "
                                "OUTPUT IS GARBAGE. Timing-only ablation.\n");
        return on;
    }();
    return v;
}

struct pxa_dq_stat { std::atomic<uint64_t> calls{0}; std::atomic<uint64_t> halves{0}; };
static pxa_dq_stat g_pxa_dq[GGML_CUDA_MAX_DEVICES];

static void pxa_dq_report() {
    uint64_t tc = 0, th = 0;
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        const uint64_t c = g_pxa_dq[i].calls.load(), h = g_pxa_dq[i].halves.load();
        if (!c) continue;
        fprintf(stderr, "PXA_DQ dev%d: %llu src0 weight dequants, %.2f G halves, %.2f GiB written\n",
                i, (unsigned long long) c, h/1e9, h*2.0/(1024.0*1024.0*1024.0));
        tc += c; th += h;
    }
    if (tc) fprintf(stderr, "PXA_DQ total: %llu dequants, %.2f G halves, %.2f GiB written to scratch\n",
                    (unsigned long long) tc, th/1e9, th*2.0/(1024.0*1024.0*1024.0));
    pxa_dqc_report_line();
}

// One persistent zero buffer per device for the ablation. Sized to the largest weight seen; the
// GEMM reads it instead of a freshly dequantized tile, so the ablation costs neither the dequant
// kernel nor a per-call memset. Never freed - the process is exiting either way, and this is not
// a shipping path.
static void * g_pxa_dq_zero[GGML_CUDA_MAX_DEVICES]     = { nullptr };
static size_t g_pxa_dq_zero_sz[GGML_CUDA_MAX_DEVICES]  = { 0 };
static std::mutex g_pxa_dq_zero_mtx;

static const half * pxa_dq_zero_buf(int device, size_t ne, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(g_pxa_dq_zero_mtx);
    const size_t need = ne*sizeof(half);
    if (g_pxa_dq_zero_sz[device] < need) {
        if (g_pxa_dq_zero[device]) cudaFree(g_pxa_dq_zero[device]);
        if (cudaMalloc(&g_pxa_dq_zero[device], need) != cudaSuccess) { g_pxa_dq_zero_sz[device] = 0; return nullptr; }
        cudaMemsetAsync(g_pxa_dq_zero[device], 0, need, stream);
        g_pxa_dq_zero_sz[device] = need;
    }
    return (const half *) g_pxa_dq_zero[device];
}

static void ggml_cuda_op_mul_mat_cublas(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    const int compute_capability = ggml_cuda_info().devices[id].cc;

    if (src0->type == GGML_TYPE_BF16 && ggml_is_contiguous(src0) && row_diff == src0->ne[1]) {

        ggml_cuda_pool_alloc<nv_bfloat16> src1_as_bf16(ctx.pool(id));
        if (src1->type != GGML_TYPE_BF16) {
            const to_bf16_cuda_t to_bf16_cuda = ggml_get_to_bf16_cuda(src1->type);
            GGML_ASSERT(to_bf16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_bf16.alloc(ne);
            to_bf16_cuda(src1_ddf_i, src1_as_bf16.get(), src1_ncols, ne10, stream);
        }
        const nv_bfloat16 * src1_ptr = src1->type == GGML_TYPE_BF16 ? (const nv_bfloat16 *) src1_ddf_i : src1_as_bf16.get();
        const nv_bfloat16 * src0_ptr = (const nv_bfloat16 *)src0_dd_i;
        ggml_cuda_pool_alloc<nv_bfloat16> dst_bf16(ctx.pool(id), row_diff*src1_ncols);

        const float alpha_f32 = 1.0f;
        const float beta_f32  = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha_f32,  src0_ptr,       CUDA_R_16BF, ne00,
                                 src1_ptr,       CUDA_R_16BF, ne10,
                    &beta_f32,   dst_bf16.get(), CUDA_R_16BF, ldc,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
        to_fp32_cuda(dst_bf16.get(), dst_dd_i, row_diff, src1_ncols, stream);
        return;
    }

#ifdef GGML_CUDA_IQK_FORCE_BF16
    if (ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && row_diff == src0->ne[1]) {
        to_bf16_cuda_t to_bf16_cuda = ggml_get_to_bf16_cuda(src0->type);
        to_bf16_cuda_t to_bf16_cuda_1 = src1->type != GGML_TYPE_BF16 ? ggml_get_to_bf16_cuda(src1->type) : nullptr;
        if (to_bf16_cuda && (src1->type == GGML_TYPE_BF16 || to_bf16_cuda_1)) {
            size_t ne = row_diff*ne00;
            ggml_cuda_pool_alloc<nv_bfloat16> src0_as_bf16(ctx.pool(id), ne);
            to_bf16_cuda(src0_dd_i, src0_as_bf16.get(), row_diff, ne00, stream);

            ggml_cuda_pool_alloc<nv_bfloat16> src1_as_bf16(ctx.pool(id));
            if (src1->type != GGML_TYPE_BF16) {
                size_t ne = src1_ncols*ne10;
                src1_as_bf16.alloc(ne);
                to_bf16_cuda_1(src1_ddf_i, src1_as_bf16.get(), src1_ncols, ne10, stream);
            }
            const nv_bfloat16 * src1_ptr = src1->type == GGML_TYPE_BF16 ? (const nv_bfloat16 *) src1_ddf_i : src1_as_bf16.get();
            const nv_bfloat16 * src0_ptr = src0_as_bf16.get();

            ggml_cuda_pool_alloc<nv_bfloat16> dst_bf16(ctx.pool(id), row_diff*src1_ncols);

            const float alpha_f32 = 1.0f;
            const float beta_f32  = 0.0f;

            CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
            CUBLAS_CHECK(
                    cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                        row_diff, src1_ncols, ne10,
                        &alpha_f32,  src0_ptr,       CUDA_R_16BF, ne00,
                        src1_ptr,       CUDA_R_16BF, ne10,
                        &beta_f32,   dst_bf16.get(), CUDA_R_16BF, ldc,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
            to_fp32_cuda(dst_bf16.get(), dst_dd_i, row_diff, src1_ncols, stream);
            return;
        }
    }
#endif

    // PXA_P100_FP16_GEMM_v1 (2026-07-15): GP100 (sm_60) has NATIVE double-rate fp16 (HFMA2/half2,
    // ~19 TF vs 9.5 TF fp32) — it was the flagship fp16 card before Volta. The old CC_VOLTA gate
    // (aimed at GP102/GP104, whose fp16 is 1/64 rate) forced P100 onto dequant->fp32 + SGEMM for
    // every quantized prompt matmul: measured 80%% of gpt-oss prefill wall (dequantize_block_mxfp4<float>
    // 30%% + maxwell_sgemm/sgemm_*_vec 49%%). fast_fp16_available() (cc>=600 && cc!=610) already knows
    // the truth — use it here: fp16 dequant (half the write traffic) + GemmEx 16F (2x FLOP rate).
    // sm_61 (1080Ti) stays excluded. Env PXA_P100_FP16_GEMM=0 rolls back to the old fp32 path.
    // PXA 0a hygiene (2026-07-22): level-aware resolver (pxa-enhance.cuh) — REFERENCE -> false so
    // PXA_REFERENCE=1 baselines on sm_60 really run the pure fp32 SGEMM reference path. Env wins.
    // PXA_F32PREC_F16GEMM (2026-08-10, default OFF): a GGML_PREC_F32 matmul over an f16/quantized
    // src0 falls through this function to dequant-to-FP32 + cublasSgemm — volta_sgemm at ~15.7
    // TFLOPS, measured 20.1 ms/call x 408 calls = 50.7%% of the Muse-Glimmer Q4_K_XL prefill wall
    // on 2x V100. Mainline serves the IDENTICAL ops with f16 INPUTS and CUBLAS_COMPUTE_32F
    // (cutlass s884gemm_f16): tensor cores, fp32 ACCUMULATION — the precision the graph asked
    // for is honored in the accumulator; only the operand load rounds to f16, which is the same
    // numeric class mainline ships for this model. Armed: PREC_F32 ops take the f16-input branch
    // below with a per-op fp32 compute type. G3-class (not bit-exact vs the fp32-operand SGEMM);
    // quality-gated before any default flip.
    static const bool pxa_f32prec_f16gemm = [](){
        const char * e = getenv("PXA_F32PREC_F16GEMM");
        const bool v = e && atoi(e) != 0;
        if (v) fprintf(stderr, "PXA_F32PREC_F16GEMM ARMED: PREC_F32 dense GEMMs run f16-input + CUBLAS_COMPUTE_32F (tensor cores, fp32 accumulate) instead of dequant-f32 + SGEMM\n");
        return v;
    }();
    const bool pxa_f32prec_take = pxa_f32prec_f16gemm && dst->op_params[0] != GGML_PREC_DEFAULT;
    // PXA_VOLTA_F16_GEMM (2026-09-02, default ON for cc == 7.0) — take the f16-input,
    // fp32-accumulate, F32-OUTPUT GemmEx form for EVERY dense GEMM on Volta, exactly as
    // mainline llama.cpp does (`prefer_f32_output = cc == GGML_CUDA_CC_VOLTA`, A/B in
    // CUDA_R_16F, CUBLAS_COMPUTE_32F, dst CUDA_R_32F).
    //
    // Two defects it closes at once on sm_70, both measured by nvprof on the 2xV100 pair
    // (Qwable-27B, 20801-token prefill, 2 reps, 2026-09-02):
    //  (a) op_params[0] ALIAS -> SGEMM. A FUSED_UP_GATE node stores the UNARY OP id in
    //      op_params[0]; the generic branch reads that slot as a ggml_prec, sees != DEFAULT
    //      and vetoes the whole f16 branch -> dequantize-to-FP32 + cublasSgemm. On the MXFP4
    //      twin of this model that is volta_sgemm_128x128_tn at 68.5 s = 51.2% of all GPU
    //      time, against 14.1 s of real tensor-core cutlass h884. It is why our MXFP4 prefill
    //      is 373/317 t/s where mainline, serving the identical file with the identical
    //      dequant+cuBLAS structure, gets 940/1195. Volta has no reason to ever run SGEMM
    //      here: the f16 operands feed HMMA and the accumulator is fp32 either way.
    //  (b) the fp16-accumulate round trip. The PREC_DEFAULT path ran COMPUTE_16F into a
    //      pool-allocated f16 dst and then converted it back with to_fp32. That conversion
    //      is convert_unary<half,float>, 1.29 s / 10846 calls = 2.1% of PXQ4 prefill GPU
    //      time at 20801, plus a pool allocation per call — and it accumulates in fp16.
    //      Mainline's Volta route writes F32 straight into dst. Strictly better: one fewer
    //      kernel, one fewer buffer, and a WIDER accumulator.
    //
    // Note (b) also fixes a latent bug: the f16 branch sized its pool buffer row_diff*ncols
    // but let cuBLAS write with leading dimension `ldc`, which is ne0 (> row_diff) on the
    // main device of a split tensor. The F32 form writes straight into dst_dd_i with ldc.
    //
    // Not bit-exact vs the old PREC_DEFAULT path (fp32 vs fp16 accumulation) and not
    // bit-exact vs the old PREC_F32 path (f16 vs f32 operands) — the accumulator is fp32 in
    // the new path either way, so this is the same numeric class mainline ships for these
    // models. Volta only; every other arch keeps its exact incumbent behaviour.
    // PXA_VOLTA_F16_GEMM=0 restores the old dispatch on sm_70.
    static const bool pxa_volta_f16_gemm = [](){
        const char * e = getenv("PXA_VOLTA_F16_GEMM");
        return !(e && atoi(e) == 0);
    }();
    const bool pxa_volta_f32out = pxa_volta_f16_gemm && compute_capability == CC_VOLTA;
    const bool pxa_fp16_gemm_ok = compute_capability >= CC_VOLTA ||
        (pxa_p100_fp16_gemm() && compute_capability < CC_VOLTA && fast_fp16_available(compute_capability));
    if (pxa_fp16_gemm_ok && (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || ggml_is_quantized(src0->type)) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] && (dst->op_params[0] == GGML_PREC_DEFAULT || pxa_f32prec_take || pxa_volta_f32out)) {
        // convert src0 and src1 to fp16, multiply as fp16, convert dst to fp32
        ggml_cuda_pool_alloc<half> src0_as_f16(ctx.pool(id));
        const half * pxa_dq_sub = nullptr;
        if (src0->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src0->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = row_diff*ne00;
            // PXA_DQ: count first, so PROF reports the work even when ABLATE removes it.
            if (pxa_dq_prof()) {
                static std::once_flag once;
                std::call_once(once, [](){ atexit(pxa_dq_report); });
                const uint64_t nc = g_pxa_dq[id].calls.fetch_add(1, std::memory_order_relaxed) + 1;
                g_pxa_dq[id].halves.fetch_add((uint64_t) ne, std::memory_order_relaxed);
                // atexit() never runs here: the bench harness stops the server with `docker rm -f`,
                // which is SIGKILL. Report periodically instead so the counters actually reach the
                // log. ~325 PXQ tensors x 10 ubatches = ~3250 dequants per 20k prefill, so this is
                // roughly one line per prefill and cannot become a firehose.
                if ((nc & 0xFFF) == 0) pxa_dq_report();
            }
            if (pxa_dq_ablate()) {
                pxa_dq_sub = pxa_dq_zero_buf(id, ne, stream);
            }
            // K8-C: if the lookahead walker already materialized this weight in the prefetch
            // arena, take it; on a miss this is a no-op and the incumbent pool path below runs
            // byte-for-byte as before.
            if (pxa_dq_sub == nullptr) {
                pxa_dq_sub = pxa_dqc_acquire(id, src0_dd_i, row_diff, ne00, (int) src0->type, stream);
            }
            if (pxa_dq_sub == nullptr) {
                src0_as_f16.alloc(ne);
                to_fp16_cuda(src0_dd_i, src0_as_f16.get(), row_diff, ne00, stream);
            }
        }
        const half * src0_ptr = src0->type == GGML_TYPE_F16 ? (const half *) src0_dd_i
                              : (pxa_dq_sub ? pxa_dq_sub : src0_as_f16.get());

        ggml_cuda_pool_alloc<half> src1_as_f16(ctx.pool(id));
        if (src1->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_cuda(src1_ddf_i, src1_as_f16.get(), src1_ncols, ne10, stream);
        }
        const half * src1_ptr = src1->type == GGML_TYPE_F16 ? (const half *) src1_ddf_i : src1_as_f16.get();

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        if (pxa_f32prec_take || pxa_volta_f32out) {
            // PREC_F32, armed: fp32 accumulate + F32 output straight into dst (no round-trip).
            // On Volta (pxa_volta_f32out) this is the route for EVERY precision — see
            // PXA_VOLTA_F16_GEMM above.
            const float alpha_f32 = 1.0f;
            const float beta_f32  = 0.0f;
            CUBLAS_CHECK(
                cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                        row_diff, src1_ncols, ne10,
                        &alpha_f32, src0_ptr, CUDA_R_16F, ne00,
                                    src1_ptr, CUDA_R_16F, ne10,
                        &beta_f32,  dst_dd_i, CUDA_R_32F, ldc,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            GGML_UNUSED(dst);
            GGML_UNUSED(src1_ddq_i);
            return;
        }
        ggml_cuda_pool_alloc<half> dst_f16(ctx.pool(id), row_diff*src1_ncols);

        const half alpha_f16 = 1.0f;
        const half beta_f16 = 0.0f;

        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha_f16, src0_ptr,       CUDA_R_16F, ne00,
                                src1_ptr,       CUDA_R_16F, ne10,
                    &beta_f16,   dst_f16.get(), CUDA_R_16F, ldc,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        to_fp32_cuda(dst_f16.get(), dst_dd_i, row_diff, src1_ncols, stream);
    } else {
        ggml_cuda_pool_alloc<float> src0_ddq_as_f32(ctx.pool(id));
        ggml_cuda_pool_alloc<float> src1_ddq_as_f32(ctx.pool(id));

        if (src0->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_cuda(src0_dd_i, src0_ddq_as_f32.get(), row_diff, ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src1->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_cuda(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols, ne10, stream);
        }

        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasSgemm(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha, src0_ddf_i,  ne00,
                            src1_ddf1_i, ne10,
                    &beta,  dst_dd_i,    ldc));
    }

    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}

static bool ggml_cuda_set_peer_access(int main_device) {
    ggml_cuda_set_device(main_device);

    bool all_enabled = true;
    for (int id_other = 0; id_other < ggml_backend_cuda_get_device_count(); ++id_other) {
        if (main_device == id_other) {
            continue;
        }

        int can_access_peer;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, main_device, id_other));
        if (can_access_peer) {
//~ #ifdef NDEBUG
            GGML_CUDA_LOG_INFO(" =========================== %s: Enabling Peer Access between Devices %d->%d\n", __func__, main_device, id_other);
//~ #endif //NDEBUG
            cudaError_t err = cudaDeviceEnablePeerAccess(id_other, 0);
            if (err != cudaErrorPeerAccessAlreadyEnabled) {
                CUDA_CHECK(err);
            } else {
                // reset the error
                (void)cudaGetLastError();
            }
        } else {
            all_enabled = false;
        }
    }
    return all_enabled;
}

static cudaError_t ggml_cuda_Memcpy2DPeerAsync(
    void * dst, int dstDevice, size_t dpitch, void * src, int srcDevice, size_t spitch, size_t width, size_t height, cudaStream_t stream) {

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
    // cudaMemcpy2DAsync may fail with copies between vmm pools of different devices
    cudaMemcpy3DPeerParms p = {};
    p.dstDevice = dstDevice;
    p.dstPtr = make_cudaPitchedPtr(dst, dpitch, dpitch, height);
    p.srcDevice = srcDevice;
    p.srcPtr = make_cudaPitchedPtr(src, spitch, spitch, height);
    p.extent = make_cudaExtent(width, height, 1);
    return cudaMemcpy3DPeerAsync(&p, stream);
#else
    // HIP does not support cudaMemcpy3DPeerAsync or vmm pools
    GGML_UNUSED(dstDevice);
    GGML_UNUSED(srcDevice);
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, stream);
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
}

static void ggml_cuda_op_mul_mat(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_cuda_op_mul_mat_t op,
    quantize_cuda_t quantize_src1) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    GGML_ASSERT(ggml_backend_buffer_is_cuda(dst->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src1->buffer));
    ggml_backend_cuda_buffer_context * src1_ctx = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
    ggml_backend_cuda_buffer_context * dst_ctx  = (ggml_backend_cuda_buffer_context *) dst->buffer->context;

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    const int64_t i02_divisor = ne12 / ne02;

    const size_t src0_rs = ggml_row_size(src0->type, ne00);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    const int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    struct dev_data {
        int cc;

        ggml_cuda_pool_alloc<char>   src0_dd_alloc;
        ggml_cuda_pool_alloc<float> src1_ddf_alloc;
        ggml_cuda_pool_alloc<char>  src1_ddq_alloc;
        ggml_cuda_pool_alloc<float>   dst_dd_alloc;

        char  *  src0_dd = nullptr;
        float * src1_ddf = nullptr; // float
        char  * src1_ddq = nullptr; // q8_1
        float *   dst_dd = nullptr;

        int64_t  row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_CUDA_MAX_DEVICES];

    int used_devices = 0;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        dev[id].cc = ggml_cuda_info().devices[id].cc;

        // by default, use all rows
        dev[id].row_low  = 0;
        dev[id].row_high = ne01;

    }

    bool quantization_done = false;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if (id != ctx.device || dev[id].row_low == dev[id].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = id == src1_ctx->device;
        const bool  dst_on_device = id == dst_ctx->device;

        ggml_cuda_set_device(id);
        cudaStream_t stream = ctx.stream(id, 0);

        if (src0_is_contiguous) {
            dev[id].src0_dd = (char *) src0->data;
        } else {
            // If src0 is not contiguous it will be copied to a temporary buffer, it may then be necessary to clear padding.
            const size_t nbytes_data    = ggml_nbytes(src0);
            const size_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            dev[id].src0_dd = dev[id].src0_dd_alloc.alloc(ctx.pool(id), nbytes_data + nbytes_padding);
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd, 0, nbytes_data + nbytes_padding, stream));
        }

        // If src0 is on a temporary compute buffer (partial offloading) there may be some padding that needs to be cleared:
        if (ne00 % MATRIX_ROW_PADDING != 0 && ggml_is_quantized(src0->type) && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE && src0->view_src == nullptr) {
            const int64_t nbytes_data    = ggml_row_size(src0->type, (dev[id].row_high - dev[id].row_low)*ne00);
            const int64_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd + nbytes_data , 0, nbytes_padding, stream));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[id].src1_ddf = (float *) src1->data;
        } else {
            dev[id].src1_ddf = dev[id].src1_ddf_alloc.alloc(ctx.pool(id), ggml_nelements(src1));
        }

        if (quantize_src1) {
            size_t src_1_ddq_size = nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs;
            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                src_1_ddq_size += get_mmq_x_max_host(dev[id].cc)*sizeof(block_q8_1_mmq);
            }
            dev[id].src1_ddq = dev[id].src1_ddq_alloc.alloc(ctx.pool(id), src_1_ddq_size);

            if (src1_on_device && (src1_is_contiguous || (src1->ne[1] == 1 && src1->ne[3] == 1 && src1->nb[0] == sizeof(float)))) {
                if (src1_is_contiguous) {
                    quantize_src1(dev[id].src1_ddf, dev[id].src1_ddq, ne10, ne11, ne12*ne13, src1_padded_col_size, src0->type, stream);
                } else {
                    //printf("Calling quantize_tensor_q8_1_cuda for %s\n", src0->name);
                    quantize_tensor_q8_1_cuda(src1, dev[id].src1_ddq, src0->type, stream);
                }
                CUDA_CHECK(cudaGetLastError());
                quantization_done = true;
            }
        }

        if (dst_on_device) {
            dev[id].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = ggml_nelements(dst);
            dev[id].dst_dd = dev[id].dst_dd_alloc.alloc(ctx.pool(id), size_dst_ddf);
        }
    }

    const int64_t src1_col_stride = ne11;
    // split-buffer src0 has data == NULL; per-device dispatch happens in the slow path below.
    const bool src0_is_split = src0->buffer && ggml_backend_buft_is_cuda_split(src0->buffer->buft);
    if (quantization_done && ne11 == 1 && ne12 > 1 && ne13 == 1 && ne02 == ne12 && ne02 == dst->ne[2] && !src0_is_split) {
        int id = ctx.device;
        char  *  src0_dd_i =  dev[id].src0_dd;
        float * src1_ddf_i = dev[id].src1_ddf;
        char  * src1_ddq_i = dev[id].src1_ddq;
        float *   dst_dd_i =   dev[id].dst_dd;
        cudaStream_t stream = ctx.stream(id, 0);
        ggml_cuda_op_mul_mat_vec_q_3D(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                dev[id].row_low, dev[id].row_high, ne11, src1_padded_col_size, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;

        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if (id != ctx.device || dev[id].row_low == dev[id].row_high) {
                continue;
            }

            const bool src1_on_device = id == src1_ctx->device;
            const bool  dst_on_device = id == dst_ctx->device;
            const int64_t row_diff = dev[id].row_high - dev[id].row_low;

            ggml_cuda_set_device(id);
            cudaStream_t stream = ctx.stream(id, is);

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                size_t src1_ddq_i_offset = i0*ne11 * src1_padded_col_size*q8_1_ts/q8_1_bs;
                if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                    src1_ddq_i_offset += src1_col_0 * sizeof(block_q8_1_mmq);
                } else {
                    src1_ddq_i_offset += src1_col_0 * src1_padded_col_size*q8_1_ts/q8_1_bs;
                }

                // for split tensors the data begins at i0 == i0_offset_low
                char  *  src0_dd_i =  dev[id].src0_dd + (i0/i02_divisor) * ne01*src0_rs;
                float * src1_ddf_i = dev[id].src1_ddf + (i0*ne11 + src1_col_0) * ne10;
                char  * src1_ddq_i = dev[id].src1_ddq +  src1_ddq_i_offset;
                float *   dst_dd_i =   dev[id].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (id == ctx.device) {
                    dst_dd_i += dev[id].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (id != ctx.device) {
                        if (quantize_src1) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                                const size_t pitch = ne11*sizeof(block_q8_1_mmq);
                                const size_t width = src1_ncols*sizeof(block_q8_1_mmq);
                                const size_t height = src1_padded_col_size/(4*QK8_1);
                                CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(src1_ddq_i, id, pitch, src1_ddq_i_source, ctx.device, pitch, width, height, stream));
                            } else {
                                CUDA_CHECK(cudaMemcpyPeerAsync(
                                    src1_ddq_i, id, src1_ddq_i_source, ctx.device, src1_ncols*src1_padded_col_size*q8_1_ts/q8_1_bs, stream));
                            }
                        } else {
                            float * src1_ddf_i_source = (float *) src1->data;
                            src1_ddf_i_source += (i0*ne11 + src1_col_0) * ne10;
                            CUDA_CHECK(cudaMemcpyPeerAsync(src1_ddf_i, id, src1_ddf_i_source, ctx.device,
                                                            src1_ncols*ne10*sizeof(float), stream));
                        }
                    }
                } else if (src1_on_device && !src1_is_contiguous) {
                    if (!quantization_done) {
                        //printf("Copying %s\n", src1->name);
                        CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                                    src1_ddf_i, src1, i03, i02, src1_col_0, src1_col_0+src1_ncols, stream));
                    }
                } else {
                    GGML_ABORT("fatal error");
                }

                if (quantize_src1 && !src1_is_contiguous && !quantization_done) {
                    //printf("Quantizing %s\n", src1->name);
                    quantize_src1(src1_ddf_i, src1_ddq_i, ne10, src1_ncols, 1, src1_padded_col_size, src0->type, stream);
                    CUDA_CHECK(cudaGetLastError());
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_divisor == 0) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_divisor, dev[id].row_low, dev[id].row_high, stream));
                }

                // do the computation
                op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[id].row_low, dev[id].row_high, src1_ncols, src1_padded_col_size, stream);
                CUDA_CHECK(cudaGetLastError());

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                    GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                    dhf_dst_i += src1_col_0*ne0;
                    CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_dd_i, src1_ncols*ne0*sizeof(float), cudaMemcpyDeviceToDevice, stream));
                }

            }
        }
    }
}

static void ggml_cuda_mul_mat_vec_p021(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src0->buffer));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t ne12 = src1->ne[2];

    cudaStream_t main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    ggml_mul_mat_p021_f16_f32_cuda(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}

static void ggml_cuda_mul_mat_vec_nc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne12 = src1->ne[2];

    cudaStream_t main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    const int64_t row_stride_x = nb01 / sizeof(half);
    const int64_t channel_stride_x = nb02 / sizeof(half);

    ggml_mul_mat_vec_nc_f16_f32_cuda(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x, main_stream);
}

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

template<ggml_type src0_type>
static void ggml_cuda_mul_mat_batched_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<src0_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(src0->type == src0_type);
    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), main_stream));

    float * dst_ddf = (float *) dst->data;
    const size_t ts_src1 = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == ts_src1);
    int64_t s11 = nb11 / ts_src1;
    int64_t s12 = nb12 / ts_src1;
    int64_t s13 = nb13 / ts_src1;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // Handle src0
    src0_ptr = (const cuda_t *) src0->data;

    // Handle src1 - convert if necessary
    if (src1->type == src0_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        // Convert src1 to target type using traits conversion functions
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_alloc.alloc(ne_src1);

        const auto convert_func = traits::get_nc_converter(src1->type);
        GGML_ASSERT(convert_func != nullptr);
        convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
        src1_ptr = src1_alloc.get();
        s11 = ne10;
        s12 = ne11*s11;
        s13 = ne12*s12;

        is_src1_cont_2 = true;
    }

    // Setup destination buffer
    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_t;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();
    const float alpha_f32 = 1.0f;
    const float beta_f32 = 0.0f;

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        if constexpr (src0_type == GGML_TYPE_F32) {
            dst_t = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_t = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    } else {
        dst_t = (char *) dst_ddf;
        cu_compute_type = CUBLAS_COMPUTE_32F;
        cu_data_type = CUDA_R_32F;
        alpha = &alpha_f32;
        beta = &beta_f32;
    }

    int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        //printf("Using cublasGemmStridedBatchedEx for %s\n", dst->name);
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
        //const int64_t smb = ne12 == 1 ? s13       : s12;
        const int64_t smb = ne12 == 1 ? nb13/nb10 : nb12/nb10;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, nb01/nb00, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11,       smb,     // strideB
                beta,     dst_t, cu_data_type,   ne0,       ne1*ne0, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        //printf("Using cublasGemmBatchedEx for %s\n", dst->name);
        //printf("    src0: %ld x %ld x %ld x %ld; %zu x %zu x %zu x %zu\n",src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
        //printf("    src1: %ld x %ld x %ld x %ld; %zu x %zu x %zu x %zu\n",src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        size_t src1_stride_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        dim3 block_dims(threads_x, threads_y);

        dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_t,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                nb02, nb03,
                (src1->type == src0_type) ? nb12 : s12*src1_stride_size,
                (src1->type == src0_type) ? nb13 : s13*src1_stride_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, nb01/nb00,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (dst->op_params[0] == GGML_PREC_DEFAULT && cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, 1, main_stream);
    }
}

static void ggml_cuda_mul_mat_batched_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F32);

    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("Unsupported type");
    }
}

// ---------------------------------------------------------------------------------------------
// G2-F3 NORMFUSE (PXA_G2_NORMFUSE=1, default OFF): a FUSED_RMS_NORM whose output feeds q8_1-
// quantized GEMV consumers emits the q8_1 of its own output as a SIDECAR in the same kernel
// (bit-identical to norm-then-quantize_q8_1, see norm.cu). Consumers (the mmvq fast-TG chain,
// FUSED_UP_GATE) look the sidecar up by producer-tensor identity and skip their quantize launch.
// The sidecar is only valid within the same graph eval (serial-checked); a miss falls back to
// the normal quantize path, so behavior is always correct.
// ---------------------------------------------------------------------------------------------
static inline bool pxa_g2_normfuse() {
    static const bool on = [](){
        const char * e = getenv("PXA_G2_NORMFUSE");
        bool v = e && atoi(e) != 0;
        if (v) fprintf(stderr, "PXA_G2_NORMFUSE: ON (G2-F3 fused rmsnorm + q8_1 sidecar producer fusion, bit-exact)\n");
        return v;
    }();
    return on;
}

struct pxa_g2_q8sc_t {
    const ggml_tensor * t = nullptr;   // producer node (sidecar = q8_1 of its output)
    const void * data     = nullptr;   // producer dst->data at emit time
    char * buf            = nullptr;   // persistent device buffer
    size_t sz             = 0;
    int64_t padded        = 0;         // ne10_padded the sidecar was laid out for
    uint64_t eval         = 0;         // graph-eval serial the sidecar was emitted in
};
static pxa_g2_q8sc_t pxa_g2_q8sc[GGML_CUDA_MAX_DEVICES];
static uint64_t pxa_g2_eval_serial = 1;

// G2-F2 QUANTFOLD (PXA_G2_QUANTFOLD=1, default OFF): same sidecar mechanism, but the emitter is
// the deltanet out-gate fused kernel (pxa_dn_rms_silu_gate_f32) -> kills the quantize_q8_1 that
// feeds linear_attn_out. Chosen over the spec's fold-into-mmvq-prologue form because each mmvq
// block traverses the FULL x vector: a per-block redundant quantize multiplies work by gridDim,
// which the busy-wall physics (g1) says is a loss; the producer-side emit removes the same
// launch + HBM round-trip with O(1) redundant compute.
static inline bool pxa_g2_quantfold() {
    static const bool on = [](){
        const char * e = getenv("PXA_G2_QUANTFOLD");
        bool v = e && atoi(e) != 0;
        if (v) fprintf(stderr, "PXA_G2_QUANTFOLD: ON (G2-F2 producer-side q8_1 sidecar from the deltanet out-gate kernel, bit-exact)\n");
        return v;
    }();
    return on;
}

static char * pxa_g2_q8_buf(int device, cudaStream_t stream, size_t need) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return nullptr;
    auto & sc = pxa_g2_q8sc[device];
    if (sc.sz >= need) return sc.buf;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &st);
    if (st != cudaStreamCaptureStatusNone) return nullptr;   // can't grow mid-capture -> decline
    if (sc.buf) cudaFree(sc.buf);
    sc.buf = nullptr; sc.sz = 0;
    if (cudaMalloc(&sc.buf, need) != cudaSuccess) { sc.buf = nullptr; cudaGetLastError(); return nullptr; }
    sc.sz = need;
    return sc.buf;
}

static const char * pxa_g2_q8_lookup(int device, const ggml_tensor * src1, int64_t ne10_padded) {
    if (!pxa_g2_normfuse()) return nullptr;
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return nullptr;
    const auto & sc = pxa_g2_q8sc[device];
    if (!sc.t || !sc.buf || sc.eval != pxa_g2_eval_serial) return nullptr;
    const ggml_tensor * base = src1->view_src ? src1->view_src : src1;
    if (base != sc.t || src1->data != sc.data) return nullptr;
    if (ggml_nrows(src1) != 1 || !ggml_is_contiguous(src1)) return nullptr;
    if (ggml_nelements(src1) != ggml_nelements(sc.t)) return nullptr;   // flat identity (reshape-safe)
    if (ne10_padded != sc.padded) return nullptr;
    return sc.buf;
}

static int ggml_cuda_mul_mat_q(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cgraph * cgraph, int node_n, bool is_gemv) {

    auto stream = ctx.stream();

    auto fusion = ctx.fusion && src1->ne[1] == 1;

    auto ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    auto nb10_padded = ne10_padded*sizeof(block_q8_1)/QK8_1;
    auto quantized_size = nb10_padded*ggml_nrows(src1);
    if (!is_gemv) {
        quantized_size += get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
    }
    ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), quantized_size);
    const char * q8v = nullptr;   // G2-F3: sidecar from a fused norm+quantize producer, if valid
    if (is_gemv) {
        q8v = pxa_g2_q8_lookup(ctx.device, src1, ne10_padded);
        if (!q8v) {
            quantize_row_q8_1_cuda((const float *)src1->data, (void *)src1_quantized.get(), src1->ne[0], src1->ne[1], src1->ne[2], ne10_padded,
                    src0->type, stream);
            CUDA_CHECK(cudaGetLastError());
            q8v = src1_quantized.get();
        }

        // The code below handles the case when Q, K, V have a bias applied after the resepctive matrix multiplication.
        // In that case the graph contains mul_mat(Q) -> mul_mat(K) -> mul_mat(V) -> add(Q) -> add(K) -> add(V)
        if (fusion && cgraph && node_n + 5 < cgraph->n_nodes &&
            cgraph->nodes[node_n+1]->op == GGML_OP_MUL_MAT &&
            cgraph->nodes[node_n+2]->op == GGML_OP_MUL_MAT &&
            ggml_cuda_mmvq_type_supported(cgraph->nodes[node_n+1]->src[0]->type) &&
            ggml_cuda_mmvq_type_supported(cgraph->nodes[node_n+2]->src[0]->type) &&
            cgraph->nodes[node_n+3]->op == GGML_OP_ADD &&
            cgraph->nodes[node_n+4]->op == GGML_OP_ADD &&
            cgraph->nodes[node_n+5]->op == GGML_OP_ADD &&
            cgraph->nodes[node_n+0] == cgraph->nodes[node_n+3]->src[0] &&
            cgraph->nodes[node_n+1] == cgraph->nodes[node_n+4]->src[0] &&
            cgraph->nodes[node_n+2] == cgraph->nodes[node_n+5]->src[0]) {
            for (int i = 0; i < 3; ++i) {
                auto src0_i = cgraph->nodes[node_n+i]->src[0];
                ggml_cuda_op_mul_mat_vec_q_biased(ctx, src0_i, src1, cgraph->nodes[node_n+i], cgraph->nodes[node_n+i+3]->src[1],
                        (const char *)src0_i->data, nullptr, q8v, (float *)cgraph->nodes[node_n+i]->data,
                        0, src0_i->ne[1], src1->ne[1], ne10_padded, stream);
                CUDA_CHECK(cudaGetLastError());
            }
            node_n += 5;
        } else if (fusion && cgraph && node_n + 1 < cgraph->n_nodes &&
                   cgraph->nodes[node_n+1]->op == GGML_OP_ADD &&
                   dst == cgraph->nodes[node_n+1]->src[0] &&
                   dst->ne[0] == cgraph->nodes[node_n+1]->src[1]->ne[0] &&
                   cgraph->nodes[node_n+1]->src[1]->type == GGML_TYPE_F32 &&
                   ggml_nrows(cgraph->nodes[node_n+1]->src[1]) == 1) {
            // We have a bias applied after the matrix multiplication and we can fuse it
            ggml_cuda_op_mul_mat_vec_q_biased(ctx, dst->src[0], src1, cgraph->nodes[node_n+1], cgraph->nodes[node_n+1]->src[1],
                 (const char *)dst->src[0]->data, nullptr, q8v, (float *)cgraph->nodes[node_n+1]->data,
                 0, dst->src[0]->ne[1], src1->ne[1], ne10_padded, stream);
            ++node_n;
        } else {
            ggml_cuda_op_mul_mat_vec_q(ctx, src0, src1, dst, (const char *)src0->data, nullptr, q8v, (float *)dst->data,
                    0, src0->ne[1], src1->ne[1], ne10_padded, stream);
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        quantize_mmq_q8_1_cuda((const float *)src1->data, src1_quantized.get(), src1->ne[0], src1->ne[1], 1, ne10_padded, src0->type, stream);
        CUDA_CHECK(cudaGetLastError());

        ggml_cuda_op_mul_mat_q(ctx, src0, src1, dst, (const char *)src0->data, nullptr, src1_quantized.get(), (float *)dst->data,
                0, src0->ne[1], src1->ne[1], ne10_padded, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    if (!cgraph) return node_n;

    while (node_n + 1 < cgraph->n_nodes) {
        dst = cgraph->nodes[node_n+1];
        if (ggml_is_empty(dst) || dst->op == GGML_OP_RESHAPE || dst->op == GGML_OP_TRANSPOSE || dst->op == GGML_OP_VIEW
                               || dst->op == GGML_OP_PERMUTE || dst->op == GGML_OP_NONE) {
            ++node_n; continue;
        }
        if (dst->op != GGML_OP_MUL_MAT || dst->src[1] != src1 || !ggml_is_quantized(dst->src[0]->type) ||
                !ggml_cuda_should_use_mmq(dst->src[0]->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[1])) break;
        // the GEMV arm below dispatches straight into mmvq; a quantized type without an mmvq
        // kernel (PXQ6/PXQ2/... tiers) would GGML_ABORT there. Stop the chain instead.
        if (is_gemv && !ggml_cuda_mmvq_type_supported(dst->src[0]->type)) break;
        if (!is_gemv && mmq_get_q8_1_ds_layout(src0->type) != mmq_get_q8_1_ds_layout(dst->src[0]->type)) break;
        if (is_gemv) {
            if (fusion && node_n + 2 < cgraph->n_nodes &&
                cgraph->nodes[node_n+2]->op == GGML_OP_ADD &&
                dst == cgraph->nodes[node_n+2]->src[0] &&
                dst->ne[0] == cgraph->nodes[node_n+2]->src[1]->ne[0] &&
                cgraph->nodes[node_n+2]->src[1]->type == GGML_TYPE_F32 &&
                ggml_nrows(cgraph->nodes[node_n+2]->src[1]) == 1) {
                // We have a bias applied after the matrix multiplication and we can fuse it
                ggml_cuda_op_mul_mat_vec_q_biased(ctx, dst->src[0], src1, cgraph->nodes[node_n+2], cgraph->nodes[node_n+2]->src[1],
                        (const char *)dst->src[0]->data, nullptr, q8v, (float *)cgraph->nodes[node_n+2]->data,
                        0, dst->src[0]->ne[1], src1->ne[1], ne10_padded, stream);
                ++node_n;
            } else {
                ggml_cuda_op_mul_mat_vec_q(ctx, dst->src[0], src1, dst, (const char *)dst->src[0]->data, nullptr, q8v,
                        (float *)dst->data, 0, dst->src[0]->ne[1], src1->ne[1], ne10_padded, stream);
            }
        } else {
            ggml_cuda_op_mul_mat_q(ctx, dst->src[0], src1, dst, (const char *)dst->src[0]->data, nullptr, src1_quantized.get(),
                    (float *)dst->data, 0, dst->src[0]->ne[1], src1->ne[1], ne10_padded, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        ++node_n;
    }

    return node_n;

}

// PXA_SPEC_1ROW (2026-07-23): one block per src1 token column (Ny<=8). The ne01==1 F32
// shared-expert gate at MTP spec-verify batch sizes (Ny=2..4) misses EVERY fast dispatch path —
// dmmv/mmvq/mmq need a quantized src0, batched-cublas needs src1->ne[2]*ne[3] > 1, and the
// ne11==1-only branches below don't fire — so it fell through to a bare `cublasSgemm`
// (per-call overhead every decode token, and cuBLAS's lazy workspace alloc can fail
// mid-inference on a near-full card as CUBLAS_STATUS_INVALID_VALUE; reported on sm_86).
// Ny==1 launches grid=1 with the identical arithmetic DAG as before (bit-identical).
template <typename src_t, int block_size = 256>
static __global__ void mul_mat_row(int n, const src_t * x, const float * y, float * z,
        const size_t nb11, const size_t nb1) {
    const float * ycol = (const float *)((const char *) y + (size_t) blockIdx.x * nb11);
    float sum = 0;
    for (int i = threadIdx.x; i < n; i += block_size) {
        float xi = ggml_cuda_cast<float, src_t>(x[i]);
        sum += xi * ycol[i];
    }
    sum = warp_reduce_sum(sum);
    if constexpr (block_size > WARP_SIZE) {
        __shared__ float tmp[block_size/WARP_SIZE];
        if (threadIdx.x % WARP_SIZE == 0) {
            tmp[threadIdx.x / WARP_SIZE] = sum;
        }
        __syncthreads();
        sum = threadIdx.x < block_size / WARP_SIZE ? tmp[threadIdx.x] : 0.0f;
        sum = warp_reduce_sum(sum);
    }
    if (threadIdx.x == 0) {
        *(float *)((char *) z + (size_t) blockIdx.x * nb1) = sum;
    }
}

// ===================== PXA_SHEXP_GATE_FUSE + PXA_GLUE_DBG (2026-08-04) =====================
// PXA_SHEXP_GATE_FUSE (default OFF): fold the shared-expert 1-row gate GEMV (mul_mat_row<256>,
// e.g. qwen35moe ffn_gate_inp_shexp) into its sole fused_mul_sigmoid consumer — ONE launch
// instead of two, per MoE layer per token (48/token on the 122B). BIT-EXACT: every block of the
// fused kernel recomputes the gate dot with arithmetic identical to mul_mat_row<256> (same
// per-thread stride-256 partials, same warp_reduce_sum butterfly, same 8-partial second
// reduction => identical bits in every block), then applies fused_mul_sigmoid_f32's exact
// expression y/(1+expf(-g)). The producer node is deferred at its own graph slot and executed
// inside the consumer's slot; the defer arms only after validating (a) the consumer is a
// broadcast FUSED_MUL_UNARY(SIGMOID) whose src0 is the gate, (b) NOTHING else reads the gate
// (bounded 64-node forward scan, views included). A deferred producer that is never consumed
// hard-aborts — this fusion cannot fail silently.
struct pxa_shexp_gate_pending_t {
    const ggml_tensor * mm_dst   = nullptr;  // deferred 1-row gate MUL_MAT node
    const ggml_tensor * consumer = nullptr;  // its sole FUSED_MUL_UNARY(SIGMOID) consumer
};
static pxa_shexp_gate_pending_t pxa_shexp_gate_pending[GGML_CUDA_MAX_DEVICES];

static inline bool pxa_shexp_gate_fuse_on() {
    static const bool on = [](){
        const char * e = getenv("PXA_SHEXP_GATE_FUSE");
        const bool v = e && atoi(e) != 0;
        if (v) fprintf(stderr, "PXA_SHEXP_GATE_FUSE: ARMED (shexp gate GEMV folded into fused_mul_sigmoid; bit-exact; unset/0 reverts)\n");
        return v;
    }();
    return on;
}

// grid = (ceil(ne0/256), rows); 256 threads. Every block recomputes the row's gate dot
// bit-identically to mul_mat_row<256>, then writes its ne0-slice of y/(1+expf(-g)).
// Block (0, row) additionally stores the gate logit to the MUL_MAT dst (same bits as
// mul_mat_row would have written) so the graph tensor stays valid for inspection.
static __global__ void __launch_bounds__(256) k_pxa_shexp_gate_fuse_f32(
        const float * __restrict__ w,        // gate weight row [n]
        const float * __restrict__ x,        // input rows      [n, rows], row stride nb11f floats
        const float * __restrict__ y,        // shexp ffn out   [ne0, rows], contiguous
        float       * __restrict__ gate_out, // MUL_MAT dst     [1, rows],  row stride nbgf floats
        float       * __restrict__ dst,      // fused output    [ne0, rows], contiguous
        const int n, const int ne0, const size_t nb11f, const size_t nbgf) {
    constexpr int block_size = 256;
    const int row = blockIdx.y;
    const float * xr = x + (size_t) row * nb11f;
    float sum = 0.0f;
    for (int ii = threadIdx.x; ii < n; ii += block_size) {
        sum += w[ii] * xr[ii];                       // mul_mat_row: xi * ycol[i], same order
    }
    sum = warp_reduce_sum(sum);
    __shared__ float tmp[block_size/WARP_SIZE];
    if (threadIdx.x % WARP_SIZE == 0) {
        tmp[threadIdx.x / WARP_SIZE] = sum;
    }
    __syncthreads();
    sum = threadIdx.x < block_size/WARP_SIZE ? tmp[threadIdx.x] : 0.0f;
    sum = warp_reduce_sum(sum);                      // all lanes of warp 0 now hold the dot
    if (threadIdx.x == 0) {
        tmp[0] = sum;
        if (blockIdx.x == 0) {
            gate_out[(size_t) row * nbgf] = sum;     // identical bits to mul_mat_row's write
        }
    }
    __syncthreads();
    const float g  = tmp[0];
    const int   i0 = blockIdx.x*block_size + (int) threadIdx.x;
    if (i0 < ne0) {
        // fused_mul_sigmoid_f32: dst[i] = y[i] / (1.0f + expf(-x[row])) — exact expression
        dst[(size_t) row * ne0 + i0] = y[(size_t) row * ne0 + i0] / (1.0f + expf(-g));
    }
}

// Producer-side gate. Called from the 1-row GEMV dispatch with the graph in hand. Validates
// EVERYTHING the consumer-side launch needs, so an armed defer can never be dropped benignly:
// either the consumer node fires it, or the next arming attempt aborts the process loudly.
static bool pxa_shexp_gate_try_defer(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, ggml_tensor * dst, const ggml_cgraph * cgraph, int node_n) {
    if (!pxa_shexp_gate_fuse_on() || !cgraph || !ctx.fusion) return false;
    if (src0->type != GGML_TYPE_F32) return false;                        // f16/bf16 stay on mul_mat_1row
    if (dst->ne[0] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) return false;
    const int64_t rows = dst->ne[1];
    if (rows < 1 || rows > 8 || src1->ne[1] != rows) return false;
    if (src1->nb[1] % sizeof(float) != 0 || dst->nb[1] % sizeof(float) != 0) return false;
    // sole consumer = a broadcast FUSED_MUL_UNARY(SIGMOID) within the next 8 nodes; the FIRST
    // reader of dst decides, any other reader shape declines
    const ggml_tensor * consumer = nullptr;
    const int lim = std::min(node_n + 9, cgraph->n_nodes);
    for (int j = node_n + 1; j < lim; ++j) {
        const ggml_tensor * nd = cgraph->nodes[j];
        bool reads = nd->view_src == dst;
        for (int s = 0; s < GGML_MAX_SRC && !reads; ++s) {
            const ggml_tensor * sp = nd->src[s];
            reads = sp == dst || (sp && sp->view_src == dst);
        }
        if (!reads) continue;
        if (nd->op == GGML_OP_FUSED_MUL_UNARY &&
            (ggml_unary_op) nd->op_params[0] == GGML_UNARY_OP_SIGMOID &&
            nd->src[0] == dst && nd->src[1] && nd->src[1] != dst) {
            consumer = nd;
        }
        break;
    }
    if (!consumer) return false;
    const ggml_tensor * cs1 = consumer->src[1];
    if (cs1->type != GGML_TYPE_F32 || consumer->type != GGML_TYPE_F32) return false;
    if (cs1->ne[0] <= 1) return false;                                    // must be the broadcast form
    if (!ggml_is_contiguous(cs1) || !ggml_is_contiguous(consumer)) return false;
    if (!ggml_are_same_shape(cs1, consumer)) return false;
    if (cs1->ne[1] != rows || cs1->ne[2] != 1 || cs1->ne[3] != 1) return false;
    // beyond the consumer, the gate must not be read again (bounded scan; the gate logit is
    // single-use by construction in every shexp-gated arch in this tree)
    const int lim2 = std::min(node_n + 65, cgraph->n_nodes);
    for (int j = node_n + 1; j < lim2; ++j) {
        const ggml_tensor * nd = cgraph->nodes[j];
        if (nd == consumer) continue;
        if (nd->view_src == dst) return false;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * sp = nd->src[s];
            if (sp == dst || (sp && sp->view_src == dst)) return false;
        }
    }
    auto & pend = pxa_shexp_gate_pending[ctx.device];
    if (pend.mm_dst != nullptr) {
        fprintf(stderr, "PXA_SHEXP_GATE_FUSE: FATAL — deferred gate GEMV '%s' was never consumed; "
                        "its validated consumer did not execute. Aborting rather than risk reading "
                        "uninitialized gate logits.\n", pend.mm_dst->name);
        GGML_ABORT("PXA_SHEXP_GATE_FUSE pending not consumed");
    }
    pend.mm_dst   = dst;
    pend.consumer = consumer;
    static bool pxa_shexp_defer_banner = false;
    if (!pxa_shexp_defer_banner) {
        pxa_shexp_defer_banner = true;
        fprintf(stderr, "PXA_SHEXP_GATE_FUSE: DEFER armed on '%s' (n=%lld rows=%lld) -> consumer '%s'\n",
                dst->name, (long long) src0->ne[0], (long long) rows, consumer->name);
    }
    return true;
}

// Consumer-side launch. Matches on exact node identity; all shapes were validated at defer time.
static bool pxa_shexp_gate_fire(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto & pend = pxa_shexp_gate_pending[ctx.device];
    if (pend.consumer != dst) return false;
    const ggml_tensor * mm = pend.mm_dst;
    const ggml_tensor * w  = mm->src[0];
    const ggml_tensor * x  = mm->src[1];
    const ggml_tensor * y  = dst->src[1];
    pend.mm_dst   = nullptr;
    pend.consumer = nullptr;
    const int     n    = (int) w->ne[0];
    const int     ne0  = (int) dst->ne[0];
    const int64_t rows = dst->ne[1];
    const dim3 grid((ne0 + 255)/256, (unsigned) rows, 1);
    k_pxa_shexp_gate_fuse_f32<<<grid, 256, 0, ctx.stream()>>>(
            (const float *) w->data, (const float *) x->data, (const float *) y->data,
            (float *) mm->data, (float *) dst->data,
            n, ne0, x->nb[1]/sizeof(float), mm->nb[1]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());
    static bool pxa_shexp_fire_banner = false;
    if (!pxa_shexp_fire_banner) {
        pxa_shexp_fire_banner = true;
        fprintf(stderr, "PXA_SHEXP_GATE_FUSE: FIRING (one launch replaces gate GEMV + fused_mul_sigmoid; n=%d ne0=%d rows=%lld)\n",
                n, ne0, (long long) rows);
    }
    return true;
}

// PXA_GLUE_DBG (2026-08-04): engagement observability for the router softmax->topk fusion.
// A lever that silently never arms reads as a null but is UNTESTED — this makes the difference
// unmissable in the same run as a benchmark: first firing of each topk_moe fusion variant prints
// a banner, and the FIRST router-shaped SOFT_MAX that fails to fuse prints the node window that
// broke the pattern. Default silent; grep 'PXA_GLUE_DBG'.
static inline bool pxa_glue_dbg_on() {
    static const bool v = [](){
        const char * e = getenv("PXA_GLUE_DBG");
        return e && atoi(e) != 0;
    }();
    return v;
}
static inline void pxa_glue_topk_note(const char * variant, const ggml_tensor * sm) {
    if (!pxa_glue_dbg_on()) return;
    static int pxa_glue_notes = 0;
    if (pxa_glue_notes < 4) {
        ++pxa_glue_notes;
        fprintf(stderr, "PXA_GLUE_DBG: topk_moe FIRING variant=%s n_expert=%lld rows=%lld\n",
                variant, (long long) sm->ne[0], (long long) sm->ne[1]);
    }
}
static inline void pxa_glue_topk_miss(const ggml_cgraph * g, int i) {
    if (!pxa_glue_dbg_on()) return;
    const ggml_tensor * sm = g->nodes[i];
    // router-shaped: small expert-count softmax over decode-ish row counts
    if (sm->ne[0] < 16 || sm->ne[0] > 512 || sm->ne[1] > 8) return;
    static int pxa_glue_misses = 0;
    if (pxa_glue_misses >= 3) return;
    ++pxa_glue_misses;
    fprintf(stderr, "PXA_GLUE_DBG: router-shaped SOFT_MAX '%s' [%lld,%lld] did NOT fuse; window:",
            sm->name, (long long) sm->ne[0], (long long) sm->ne[1]);
    const int wlim = std::min(i + 9, g->n_nodes);
    for (int j = i; j < wlim; ++j) {
        fprintf(stderr, " %s", ggml_op_name(g->nodes[j]->op));
    }
    fprintf(stderr, "\n");
}
// =================== end PXA_SHEXP_GATE_FUSE + PXA_GLUE_DBG =====================

static void mul_mat_1row(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_backend_cuda_context & ctx) {
    constexpr int kBlockSize = 256;
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    const int ncols = (int) src1->ne[1];
    if (src0->type == GGML_TYPE_F16) {
        mul_mat_row<<<ncols, kBlockSize, 0, ctx.stream()>>>((int)src0->ne[0], (const half *)src0->data, (const float *)src1->data, (float *)dst->data, src1->nb[1], dst->nb[1]);
    }
    else if (src0->type == GGML_TYPE_BF16) {
        mul_mat_row<<<ncols, kBlockSize, 0, ctx.stream()>>>((int)src0->ne[0], (const nv_bfloat16 *)src0->data, (const float *)src1->data, (float *)dst->data, src1->nb[1], dst->nb[1]);
    }
    else if (src0->type == GGML_TYPE_F32) {
        mul_mat_row<<<ncols, kBlockSize, 0, ctx.stream()>>>((int)src0->ne[0], (const float *)src0->data, (const float *)src1->data, (float *)dst->data, src1->nb[1], dst->nb[1]);
    }
    else {
        GGML_ABORT("Fatal error");
    }
}

// PXA_ROUTER_FUSE B3 phase-1 (2026-07-22): the MoE router-logits GEMV (`ffn_gate_inp`, kept F32
// for precision — n_expert x n_embd against a single decode token) misses EVERY fast dispatch
// path above: dmmv/mmvq/mmq all require a quantized src0, and the F32 batched-cublas branch
// requires src1->ne[2]*ne[3] > 1 (a real batch dim). A plain F32 x F32, ne11==1 GEMV therefore
// falls all the way through to the generic `ggml_cuda_op_mul_mat_cublas`, whose fp16-GEMM branch
// also requires src0 to be F16/BF16/quantized — so an F32 src0 lands on a bare `cublasSgemm`
// call. Measured (PXA_PROFILE): ~450-480us/call on P100/V100 for what is a
// ~1M-FLOP GEMV (256 rows x 2048 K) that should cost low single-digit us — the wall is cuBLAS's
// per-call launch/workspace overhead for a GEMM shape it is not built for (N=1), paid on EVERY
// decode token, per MoE layer. Fix: a dedicated warp-per-output-row GEMV kernel — sequential
// per-thread partial sums + one warp-shuffle reduction, same "row dot x" summation shape a naive
// reference implementation would use, so top-1/top-k expert *selection* is unaffected (logits
// differ from cuBLAS's blocked/tiled reduction only at float ULP — gated on expert-id-stream
// equality, not bit-for-bit logits, matching the spec's BX-for-fp32-math-variant call).
// Restricted to the router's shape family (F32/F32/F32, ne11==1, no batch, small-ish output row
// count) so it can never intercept a real dense/attention GEMM. Default OFF pending fair-battle.
static __global__ void k_pxa_router_gemv_f32(
        const float * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int ne00, const size_t nb01) {
    const int row = blockIdx.x;
    const float * wrow = (const float *)((const char *) w + (size_t) row * nb01);
    float sum = 0.0f;
    for (int k = threadIdx.x; k < ne00; k += WARP_SIZE) {
        sum += wrow[k] * x[k];
    }
#pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, off);
    }
    if (threadIdx.x == 0) {
        y[row] = sum;
    }
}

// ROUTER_FUSE mode 3 (2026-08-03): the sm_60-shaped router GEMV. The mode-1/2 kernel above is
// ONE WARP per expert row — on a P100 that is 256 lonely warps with a 96-step serial FMA chain
// each, and it measured a 1.6% KILL vs cuBLAS on sm_60. That verdict was about the kernel
// shape, not the idea: cuBLAS gemv2T still burns 25.7 us/call (48 calls/token profiled on the
// qwen35moe 122B) for a 3 MB read whose BW floor is ~4 us. This form is 128 threads per row
// with float4 loads and a warp/smem tree — the same shape the F16 small-R GEMV ships with.
static __global__ void __launch_bounds__(128) k_pxa_router_gemv_f32_v2(
        const float * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int ne00, const size_t nb01) {
    const int row = blockIdx.x;
    const float4 * w4 = (const float4 *)((const char *) w + (size_t) row * nb01);
    const float4 * x4 = (const float4 *) x;
    const int k4 = ne00/4;
    float sum = 0.0f;
    for (int k = threadIdx.x; k < k4; k += 128) {
        const float4 a = w4[k];
        const float4 b = x4[k];
        sum += (a.x*b.x + a.y*b.y) + (a.z*b.z + a.w*b.w);
    }
    __shared__ float warpsum[4];
#pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, off);
    }
    if ((threadIdx.x & (WARP_SIZE-1)) == 0) warpsum[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        y[row] = (warpsum[0] + warpsum[1]) + (warpsum[2] + warpsum[3]);
    }
}

// PXA_F32_GEMV_NY (2026-08-30): the hc_inject class — a TINY f32 GEMV ([K=hc_dim, R=hc] with
// R=4 on qwen4exp) that cuBLAS prices at ~147us/call for a 160KB read (~300x off roofline;
// PXA_PROFILE named it 11.4% of ALL GPU time under MTP verify, 2.5% at plain decode, because
// the router-fuse path declines ny>1 and defaults off on sm_60). One 128-thread block per
// (row, token): float4 loads, the SAME canonical warp/smem fold as k_pxa_router_gemv_f32_v2.
// Reduction order differs from cuBLAS -> behavior-gauntlet gated like PXA_F16_GEMV, not
// sha-gated. Default OFF (PXA_F32_GEMV_NY=1 arms; banner on first fire).
static __global__ void __launch_bounds__(128) k_pxa_gemv_f32_ny(
        const float * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int ne00, const size_t nb01, const size_t xs_tok, const size_t ys_tok) {
    const int row = blockIdx.x;
    const int tok = blockIdx.y;
    const float4 * w4 = (const float4 *)((const char *) w + (size_t) row * nb01);
    const float4 * x4 = (const float4 *)((const char *) x + (size_t) tok * xs_tok);
    const int k4 = ne00/4;
    float sum = 0.0f;
    for (int k = threadIdx.x; k < k4; k += 128) {
        const float4 a = w4[k];
        const float4 b = x4[k];
        sum += (a.x*b.x + a.y*b.y) + (a.z*b.z + a.w*b.w);
    }
    __shared__ float warpsum[4];
#pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, off);
    }
    if ((threadIdx.x & (WARP_SIZE-1)) == 0) warpsum[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        ((float *)((char *) y + (size_t) tok * ys_tok))[row] = (warpsum[0] + warpsum[1]) + (warpsum[2] + warpsum[3]);
    }
}

static bool ggml_cuda_small_gemv_f32_ny(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const int on = [] {
        const char * e = getenv("PXA_F32_GEMV_NY");
        const int v = e ? atoi(e) : 0;
        if (v) fprintf(stderr, "PXA_F32_GEMV_NY: armed (tiny f32 GEMV, hc_inject class; "
                               "an A/B without the in-run ENGAGED banner is void)\n");
        return v;
    }();
    if (!on) return false;
    if (src0->buffer && ggml_backend_buffer_is_cuda_split(src0->buffer)) return false;
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) return false;
    if (ggml_is_transposed(src0) || ggml_is_transposed(src1)) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1) return false;
    if (src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    // the tiny-R family: a handful of output rows over a wide K. R<=64 can never be a real
    // projection (those are n_embd/hc_dim/vocab sized); K>=2048 keeps it off degenerate nodes.
    if (src0->ne[1] < 2 || src0->ne[1] > 64) return false;
    if (src0->ne[0] < 2048) return false;
    if (src1->ne[1] < 1 || src1->ne[1] > 8) return false;
    if (src0->ne[0] % 4 != 0 || (src0->nb[1] % 16) != 0 || (src1->nb[1] % 16) != 0) return false;
    if ((((uintptr_t) src0->data) % 16) != 0 || (((uintptr_t) src1->data) % 16) != 0) return false;
    static std::atomic<bool> fired{false};
    if (!fired.exchange(true)) {
        fprintf(stderr, "PXA_F32_GEMV_NY: ENGAGED (%dx%d, ny=%d)\n",
                (int) src0->ne[0], (int) src0->ne[1], (int) src1->ne[1]);
    }
    dim3 grid((unsigned) src0->ne[1], (unsigned) src1->ne[1]);
    k_pxa_gemv_f32_ny<<<grid, 128, 0, ctx.stream()>>>(
        (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
        (int) src0->ne[0], src0->nb[1], src1->nb[1], dst->nb[1]);
    return true;
}

// PXA_F16_GEMV (2026-07-26): small-R F16 GEMV for ne11==1 decode nodes. Motivation: the rev-2
// backbone table promotes the per-head attn_gate (ne[1] = n_head, 48..64 on the Laguna family)
// to F16, and an F16 x F32 ne11==1 GEMV misses every fast dispatch path (dmmv/mmvq need a
// quantized src0; the batched-cublas branch needs a real batch dim), landing on the generic
// cuBLAS chain: convert x f32->f16 + gemmSN_TN_kernel_half + convert dst f16->f32 + a cpy —
// measured (nvprof, Laguna-XS rev-2, 2xP100) ~31.5us + 3 helper kernels per call, 40 calls per
// decode token = ~1.5 ms/token for what is a ~0.3 MB tensor read. This kernel does the row-dot
// directly: f16 weights x f32 activations with fp32 per-thread partials + fixed warp/smem tree
// (4 warps per row). NOTE the cuBLAS chain rounds x AND the result through f16; this path keeps
// both in f32, so it is numerically DIFFERENT (strictly tighter) — behavior-gauntlet gated, not
// sha-gated. Restricted to R <= 512 so it can never intercept a real dense projection.
// PXA_F16_GEMV=0 rolls back to the cuBLAS chain.
static __global__ void __launch_bounds__(128) k_pxa_gemv_f16(
        const half * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int ne00, const size_t nb01) {
    const int row = blockIdx.x;
    const half2  * w2 = (const half2  *)((const char *) w + (size_t) row * nb01);
    const float2 * x2 = (const float2 *)x;
    const int k2max = ne00/2;
    float sum = 0.0f;
    for (int k = threadIdx.x; k < k2max; k += 128) {
        const half2  hw = w2[k];
        const float2 fx = x2[k];
        sum += __low2float(hw)*fx.x + __high2float(hw)*fx.y;
    }
    __shared__ float warpsum[4];
#pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, off);
    }
    if ((threadIdx.x & (WARP_SIZE-1)) == 0) warpsum[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        y[row] = (warpsum[0] + warpsum[1]) + (warpsum[2] + warpsum[3]);
    }
}

// =================================================================================================
// PXA_GEMV_RPB / PXA_GEMV_NWARPS (2026-09-01) — rows-per-block and K-sized warp count for the
// small-R F16 decode GEMV above. Technique translated from the shinbunbun sm_60 set
// (`mmvq-rows-per-block-sm60`, `mmvq-moe-rows-sm60`, `mmvq-nwarps-small-k-sm60`), which target
// stock MMVQ — a path this fork never executes on a PXQU model. The two mechanisms do map:
//
//  (a) ROWS PER BLOCK. k_pxa_gemv_f16 gives one 128-thread block to ONE output row, so the
//      activation row x is re-read from L2 once per output row. At the live hc_down shape
//      (K=10240, R=320) that is 320 x 40 KB = 12.8 MB of L2 reads against 6.55 MB of weight
//      DRAM traffic: the activation, not the weight, is the majority of the load issue.  Giving
//      one block RPB rows makes ONE x load feed RPB weight rows, exactly the reuse shinbunbun
//      measured as +15%/+23% at rows_per_block 2/4 on GP100.
//
//      BIT-IDENTICAL. Thread t still accumulates k = t, t+nthr, ... for every row it owns, and
//      the warp butterfly plus the fixed (w0+w1)+(w2+w3) fold are unchanged, so each row's fp32
//      summation order is byte-for-byte what RPB=1 produces. Only the block->row map changes.
//
//  (b) WARP COUNT vs K. The block is 4 warps wide whatever K is. A thread only enters the loop
//      if its lane index is below ne00/2, so a row with K < 256 leaves warps completely idle for
//      the life of the kernel — shinbunbun's `calc_nwarps` observation, and it applies verbatim
//      here because our block width is likewise a constant. NOT bit-identical: fewer warps means
//      a different partition of K across threads and therefore a different partial-sum tree.
//
// Both default to the stock geometry (RPB=1, 4 warps), which dispatches to the ORIGINAL kernel
// above, untouched. PXA_GEMV_RPB=-1 arms the per-shape table; a positive value pins it (honoured
// only if it tiles R exactly — like the wide kernel, there is no ragged path). PXA_GEMV_NWARPS=-1
// auto-sizes to K; a positive value pins it.
// =================================================================================================
template <int NW, int RPB>
static __global__ void __launch_bounds__(NW*WARP_SIZE) k_pxa_gemv_f16_rpb(
        const half * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int ne00, const size_t nb01) {
    const int row0 = blockIdx.x*RPB;
    const half2  * __restrict__ w2 = (const half2  *)((const char *) w + (size_t) row0 * nb01);
    const float2 * __restrict__ x2 = (const float2 *)x;
    const int k2max  = ne00/2;
    const int rstep2 = (int)(nb01/sizeof(half2));

    float sum[RPB];
#pragma unroll
    for (int r = 0; r < RPB; ++r) sum[r] = 0.0f;

    for (int k = threadIdx.x; k < k2max; k += NW*WARP_SIZE) {
        const float2 fx = x2[k];            // one activation load feeds all RPB rows
#pragma unroll
        for (int r = 0; r < RPB; ++r) {
            const half2 hw = w2[k + r*rstep2];
            sum[r] += __low2float(hw)*fx.x + __high2float(hw)*fx.y;
        }
    }

    __shared__ float warpsum[NW*RPB];
#pragma unroll
    for (int r = 0; r < RPB; ++r) {
#pragma unroll
        for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
            sum[r] += __shfl_down_sync(0xffffffff, sum[r], off);
        }
        if ((threadIdx.x & (WARP_SIZE-1)) == 0) warpsum[r*NW + (threadIdx.x >> 5)] = sum[r];
    }
    __syncthreads();
    if (threadIdx.x < RPB) {
        const float * ws = warpsum + threadIdx.x*NW;
        // Same fold shape as k_pxa_gemv_f16, so NW == 4 reproduces its result exactly.
        float acc;
        if      (NW == 4) acc = (ws[0] + ws[1]) + (ws[2] + ws[3]);
        else if (NW == 2) acc =  ws[0] + ws[1];
        else              acc =  ws[0];
        y[row0 + threadIdx.x] = acc;
    }
}

// -1 arms the per-shape table, 0/unset is stock, >0 pins.
static inline int pxa_gemv_rpb_env() {
    static const int v = [] {
        const char * e = getenv("PXA_GEMV_RPB");
        const int n = e ? atoi(e) : 0;
        if (n) fprintf(stderr, "PXA_GEMV_RPB: armed (%s) — small-R F16 decode GEMV rows per block; "
                               "bit-identical to the stock geometry; PXA_GEMV_RPB=0 reverts\n",
                       n < 0 ? "per-shape table" : "pinned");
        return n;
    }();
    return v;
}

static inline int pxa_gemv_nwarps_env() {
    static const int v = [] {
        const char * e = getenv("PXA_GEMV_NWARPS");
        const int n = e ? atoi(e) : 0;
        if (n) fprintf(stderr, "PXA_GEMV_NWARPS: armed (%s) — small-R F16 decode GEMV block width; "
                               "K is partitioned differently, so this is NOT bit-identical; "
                               "PXA_GEMV_NWARPS=0 reverts\n", n < 0 ? "sized to K" : "pinned");
        return n;
    }();
    return v;
}

// Per-shape rows-per-block table. Deliberately conservative: the reuse only pays when the
// activation row is large enough to dominate the block's load issue, which on this arch means a
// K in the thousands; below that the block is already short-lived and a bigger RPB just cuts the
// grid. Entries are hypotheses to be priced by microbench, not measured defaults — which is why
// nothing here is reachable without PXA_GEMV_RPB=-1.
static inline int pxa_gemv_rpb_table(int64_t K, int64_t R) {
    if (K >= 4096 && R % 4 == 0) return 4;   // hc_*_down class: K=10240, R=320
    if (K >= 4096 && R % 2 == 0) return 2;
    if (K >= 1024 && R % 2 == 0) return 2;   // attn_gate class: K=n_embd, R=n_head
    return 1;
}

// Returns true if it fully handled the mul_mat (caller should return immediately).
static bool ggml_cuda_small_gemv_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const bool on = [] {
        const char * e = getenv("PXA_F16_GEMV");
        return !(e && atoi(e) == 0);
    }();
    if (!on) return false;
    if (src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (src0->buffer && ggml_backend_buffer_is_cuda_split(src0->buffer)) return false;
    if (ggml_is_transposed(src0) || ggml_is_transposed(src1)) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1) return false;
    if (src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (src0->ne[1] < 2 || src0->ne[1] > 512) return false;   // small-R family; ne01==1 stays with mul_mat_1row
    if (src0->ne[0] % 2 != 0) return false;                    // half2 loads
    if (src0->nb[0] != sizeof(half) || src1->nb[0] != sizeof(float)) return false;
    if (src0->nb[1] % 4 != 0) return false;                    // half2 row alignment
    if (!ggml_is_contiguous(src1)) return false;
    if (!ggml_is_contiguous(dst) || dst->nb[0] != sizeof(float)) return false;  // kernel writes packed y

    // PXA_GEMV_RPB / PXA_GEMV_NWARPS geometry selection (see the block above). Anything that does
    // not tile R exactly, or that has no instantiation, silently falls back to the stock launch.
    const int64_t K = src0->ne[0], R = src0->ne[1];
    const int rpb_env = pxa_gemv_rpb_env();
    int rpb = 1;
    if (rpb_env < 0)      rpb = pxa_gemv_rpb_table(K, R);
    else if (rpb_env > 0) rpb = rpb_env;
    if (rpb != 1 && rpb != 2 && rpb != 4) rpb = 1;
    if (R % rpb != 0) rpb = 1;

    const int nw_env = pxa_gemv_nwarps_env();
    int nw = 4;
    if (nw_env < 0) {
        // Just enough warps to cover K in one pass over the half2 row, clamped to {1,2,4}.
        const int64_t k2 = K/2;
        nw = k2 <= WARP_SIZE ? 1 : (k2 <= 2*WARP_SIZE ? 2 : 4);
    } else if (nw_env == 1 || nw_env == 2 || nw_env == 4) {
        nw = nw_env;
    }

    if (rpb == 1 && nw == 4) {
        k_pxa_gemv_f16<<<(unsigned)R, 128, 0, ctx.stream()>>>(
            (const half *) src0->data, (const float *) src1->data, (float *) dst->data,
            (int) K, src0->nb[1]);
        return true;
    }

    static std::atomic<bool> geom_fired{false};
    if (!geom_fired.exchange(true)) {
        fprintf(stderr, "PXA_GEMV geometry: ENGAGED (K=%d R=%d, nwarps=%d rows_per_block=%d, grid=%d)\n",
                (int) K, (int) R, nw, rpb, (int)(R/rpb));
    }

    const unsigned grid = (unsigned)(R/rpb);
    const half  * w = (const half  *) src0->data;
    const float * x = (const float *) src1->data;
    float       * y = (float *) dst->data;
#define PXA_GEMV_RPB_CASE(NWV, RPBV) \
    if (nw == NWV && rpb == RPBV) { \
        k_pxa_gemv_f16_rpb<NWV, RPBV><<<grid, NWV*WARP_SIZE, 0, ctx.stream()>>>(w, x, y, (int) K, src0->nb[1]); \
        return true; \
    }
    PXA_GEMV_RPB_CASE(4, 2) PXA_GEMV_RPB_CASE(4, 4)
    PXA_GEMV_RPB_CASE(2, 1) PXA_GEMV_RPB_CASE(2, 2) PXA_GEMV_RPB_CASE(2, 4)
    PXA_GEMV_RPB_CASE(1, 1) PXA_GEMV_RPB_CASE(1, 2) PXA_GEMV_RPB_CASE(1, 4)
#undef PXA_GEMV_RPB_CASE

    k_pxa_gemv_f16<<<(unsigned)R, 128, 0, ctx.stream()>>>(
        (const half *) src0->data, (const float *) src1->data, (float *) dst->data,
        (int) K, src0->nb[1]);
    return true;
}

// Small-K / large-R F16 decode GEMV: the TRANSPOSE of the k_pxa_gemv_f16 shape family above,
// and the reason it cannot just reuse it. The hyper-connection up-projection is [K=320,
// R=10240]: R > 512 fails the small-R gate, so it lands on ggml_cuda_op_mul_mat_cublas, which
// for this shape is a THREE-kernel chain (convert x f32->f16, GemmEx COMPUTE_16F, convert dst
// f16->f32). nsys prices that chain at 3.81 ms/token over 97 calls -- 45.7us/call on a P100
// for a 6.55 MB weight read, ~60x off the memory roofline. The IDENTICAL work in the down
// direction, on k_pxa_gemv_f16, costs 19.2us/call. That 2.4x gap is pure dispatch loss.
//
// The R cap in the small-R gate is deliberately NOT raised. Its documented job is to never
// intercept a real dense projection, and raising it would admit exactly that. The safety
// argument here runs from the other side: K <= 512. Every real projection in this arch
// consumes a hidden state or an hc-wide stream, so its K is n_embd (2560) or hc_dim (10240)
// and cannot enter whatever its R. The two gates are therefore disjoint by construction.
//
// GEOMETRY, and why it is NOT a transposed port of k_pxa_gemv_f16. That kernel gives one
// 128-thread block to each row, which suits K=10240 (40 half2 iterations per thread). Applied
// to K=320 the same shape gives each thread 1.25 iterations and launches 10240 blocks to do
// it, with most of the block idling in a reduction tree. Here a WARP owns a row: the whole
// reduction is one shfl butterfly, no smem tree and no __syncthreads on the row path. K=320
// halves = 640 B = exactly 5 x 128 B, and nb01 = 640 keeps every row start 128 B aligned, so
// a warp's half2 loads are 5 perfectly aligned fully-coalesced transactions with zero
// over-fetch. This shape could not suit a warp better if it had been designed for it.
//
// RPW (rows per warp) is the one knob a memory-bound kernel has: it multiplies the loads in
// flight per warp, and lets one pass over x feed several rows, without disturbing that
// coalescing pattern at all. 16-byte loads were considered and rejected -- 640 B per row is
// not a multiple of 32 lanes x 16 B, so a uint4 form idles 20% of the lanes for no extra
// bandwidth, while half2 is already exactly balanced.
//
// x is staged in shared memory. It is only 1.28 KB, but EVERY warp reads all of it, and the
// 6.55 MB of weight streaming through the unified L1 would evict it continuously. That costs
// one __syncthreads per block, none per row.
//
// Numerics: fp32 accumulate, weights widened before the FMA, x never rounded to f16 -- the
// same policy as k_pxa_gemv_f16, and strictly TIGHTER than the cuBLAS chain it replaces,
// which rounds both x and the result through f16. So it is not bit-identical by construction:
// behavior-gauntlet gated, not sha-gated. Checked standalone against an fp64 CPU reference
// at the live shape: max ABSOLUTE error 6.05e-07 against a max |y| of 6.71, i.e. 9e-08
// relative -- plain fp32 round-off. (Do not read the max RELATIVE error per row: with 10240
// random rows one of them lands near zero and reports 5e-03 off a true value of 4.2e-05.)
// PXA_F16_GEMV_WIDE=0 rolls back to the cuBLAS chain.
#define PXA_WIDE_WARPS 8    // warps per block -> blockDim 256
#define PXA_WIDE_KMAX  512  // K bound; also the shared stage, in floats

template <int RPW>
static __global__ void __launch_bounds__(PXA_WIDE_WARPS*WARP_SIZE) k_pxa_gemv_f16_wide(
        const half * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int ne00, const size_t nb01) {
    // __align__(16) is load-bearing: the row loop reads this stage as float2, and a bare
    // __shared__ float[] is only guaranteed 4-byte aligned. nvcc happens to over-align it
    // today, so dropping this would work until it silently did not.
    __shared__ __align__(16) float sx[PXA_WIDE_KMAX];
    for (int i = threadIdx.x; i < ne00; i += PXA_WIDE_WARPS*WARP_SIZE) {
        sx[i] = x[i];
    }
    __syncthreads();

    const int lane = threadIdx.x & (WARP_SIZE-1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int row0 = (blockIdx.x*PXA_WIDE_WARPS + warp)*RPW;

    // No row bounds check: the gate admits only ne01 % (PXA_WIDE_WARPS*RPW) == 0, so every
    // warp owns RPW real rows. That keeps the full 0xffffffff shuffle mask unconditionally
    // legal -- an early return here would leave the butterfly with absent lanes. The price is
    // declining a ragged R, which the RPW search below simply falls out of.
    const half2  * __restrict__ w2  = (const half2  *)((const char *) w + (size_t) row0 * nb01);
    const float2 * __restrict__ sx2 = (const float2 *) sx;
    const int k2max  = ne00/2;
    const int rstep2 = (int)(nb01/sizeof(half2));

    float sum[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        sum[r] = 0.0f;
    }

    for (int k = lane; k < k2max; k += WARP_SIZE) {
        const float2 fx = sx2[k];   // one smem read feeds all RPW rows
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            const half2 hw = w2[k + r*rstep2];
            sum[r] += __low2float(hw)*fx.x + __high2float(hw)*fx.y;
        }
    }

#pragma unroll
    for (int r = 0; r < RPW; ++r) {
#pragma unroll
        for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
            sum[r] += __shfl_down_sync(0xffffffff, sum[r], off);
        }
        if (lane == 0) y[row0 + r] = sum[r];
    }
}

// Returns true if it fully handled the mul_mat (caller should return immediately).
static bool ggml_cuda_wide_gemv_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const bool on = [] {
        const char * e = getenv("PXA_F16_GEMV_WIDE");
        return !(e && atoi(e) == 0);
    }();
    if (!on) return false;
    // This preamble is a deliberate copy of ggml_cuda_small_gemv_f16's rather than a shared
    // helper: the two gates own opposite shape families and must be free to drift apart, and
    // this one must never silently widen because that one was edited.
    if (src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (src0->buffer && ggml_backend_buffer_is_cuda_split(src0->buffer)) return false;
    if (ggml_is_transposed(src0) || ggml_is_transposed(src1)) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1) return false;
    if (src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (src0->ne[0] < 64 || src0->ne[0] > PXA_WIDE_KMAX) return false;  // small-K: THE wall
    if (src0->ne[1] <= 512) return false;                      // large-R: what small-R declines
    if (src0->ne[0] % 2 != 0) return false;                    // half2 loads
    if (src0->nb[0] != sizeof(half) || src1->nb[0] != sizeof(float)) return false;
    if (src0->nb[1] % sizeof(half2) != 0) return false;        // half2 row stride
    if (!ggml_is_contiguous(src1)) return false;
    if (!ggml_is_contiguous(dst) || dst->nb[0] != sizeof(float)) return false;  // kernel writes packed y

    // RPW preference order, measured standalone at the live shape (K=320, R=10240, 6.55 MB)
    // on an idle card of each type, 200 iterations after a 20-iteration warmup:
    //            P100                    V100
    //   RPW=1    19.29us  340 GB/s       9.70us  676 GB/s
    //   RPW=2    16.33us  401 GB/s       8.01us  818 GB/s   <- best on BOTH
    //   RPW=4    16.49us  398 GB/s       9.76us  671 GB/s
    // RPW=2 wins because it doubles the loads in flight per warp while still leaving 640
    // blocks, i.e. enough to fill both cards; RPW=4 halves the grid to 320 and starts losing
    // more to a partial wave than it gains in memory-level parallelism, which bites the
    // 80-SM V100 harder than the 56-SM P100. The cuBLAS chain it replaces is 45.7us/call on
    // a P100 and 13.5us/call on a V100 for the identical work.
    // PXA_F16_GEMV_WIDE_RPW pins it for A/B; honoured only if it also tiles R exactly,
    // because the kernel has no ragged path.
    static const int rpw_env = [] {
        const char * e = getenv("PXA_F16_GEMV_WIDE_RPW");
        return e ? atoi(e) : 0;
    }();
    const int64_t R = src0->ne[1];
    const auto tiles = [R](int rpw) { return rpw > 0 && R % (int64_t)(PXA_WIDE_WARPS*rpw) == 0; };
    int rpw = 0;
    if      (tiles(rpw_env)) rpw = rpw_env;
    else if (tiles(2))       rpw = 2;
    else if (tiles(4))       rpw = 4;
    else if (tiles(1))       rpw = 1;
    else return false;

    const unsigned grid  = (unsigned)(R / (PXA_WIDE_WARPS*rpw));
    const half  * w = (const half  *) src0->data;
    const float * x = (const float *) src1->data;
    float       * y = (float *) dst->data;
    switch (rpw) {
        case 4: k_pxa_gemv_f16_wide<4><<<grid, PXA_WIDE_WARPS*WARP_SIZE, 0, ctx.stream()>>>(w, x, y, (int) src0->ne[0], src0->nb[1]); break;
        case 2: k_pxa_gemv_f16_wide<2><<<grid, PXA_WIDE_WARPS*WARP_SIZE, 0, ctx.stream()>>>(w, x, y, (int) src0->ne[0], src0->nb[1]); break;
        default: k_pxa_gemv_f16_wide<1><<<grid, PXA_WIDE_WARPS*WARP_SIZE, 0, ctx.stream()>>>(w, x, y, (int) src0->ne[0], src0->nb[1]); break;
    }
    return true;
}

// Returns true if it fully handled the mul_mat (caller should return immediately).
// Engagement counters. The RATIO is the assertion, not the banner: a one-shot bool cannot
// distinguish "fired once" from "fires every token". On a clean np1 non-speculative run
// expect fired ~ n_moe_layers * n_tokens. The router GEMV gate requires ne11==1, so any
// drafter (mtp or ngram) submits 1+n_draft and declines here -- fired stalls while
// declined_ne11 climbs. (NB: the PXQ codec paths carry tokens on ne[2] and are NOT affected.)
static std::atomic<uint64_t> g_pxa_rf_fired{0};
static std::atomic<uint64_t> g_pxa_rf_declined_ne11{0};
static void pxa_rf_dbg(const char * what) {
    static const bool dbg = getenv("PXA_ROUTER_FUSE_DBG") != nullptr;
    if (!dbg) return;
    const unsigned long long f = g_pxa_rf_fired.load(), d = g_pxa_rf_declined_ne11.load();
    if (((f + d) % 512ULL) == 0ULL)
        fprintf(stderr, "PXA_ROUTER_FUSE_DBG: %s fired=%llu declined_ne11=%llu\n", what, f, d);
}
static void pxa_rf_fired_banner(int mode, int ne00, int ne01) {
    static std::atomic<bool> logged{false};
    bool exp = false;
    if (logged.compare_exchange_strong(exp, true))
        fprintf(stderr, "PXA_ROUTER_FUSE: mode %d FIRING (%dx%d f32; PXA_ROUTER_FUSE=0 reverts to cuBLAS)\n",
                mode, ne00, ne01);
    g_pxa_rf_fired++; pxa_rf_dbg("fire");
}

// Reducer block shape. Both PXQ reducers are pure per-thread row maps, so this is
// bit-identical at any block size; it only changes how many SMs the launch reaches.
// Default 256 == today's behaviour byte-for-byte. 64 spreads the dense reduce from
// 12 blocks to 48 on a 56-SM chip. Measure before flipping the default.
static inline int pxa_pxq_reduce_blk() {
    static const int b = [](){
        const char * e = getenv("PXA_PXQ_REDUCE_BLK");
        int v = e ? atoi(e) : 256;
        if (v != 32 && v != 64 && v != 128 && v != 256) {
            fprintf(stderr, "PXA: PXA_PXQ_REDUCE_BLK=%d invalid (want 32/64/128/256) -- using 256\n", v);
            v = 256;
        }
        if (e) fprintf(stderr, "PXA_PXQ_REDUCE_BLK: reducer blockDim.x = %d\n", v);
        return v;
    }();
    return b;
}

static bool ggml_cuda_router_gemv_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // Per-ARCH gate (resolver in pxa-enhance.cuh, INT8_PREFILL-pattern): ENHANCE auto-enables
    // on cc==700 ONLY (+5.1..7.0% decode measured sm_70; +1.6% KILL sm_60, Pascal stays off).
    // Env always wins: PXA_ROUTER_FUSE=0 forces OFF at any level, =1 the sm_70 ship gate,
    // =2 TEST all-arch. REFERENCE/DEFAULT: off unless env-forced.
    if (!pxa_router_fuse_on(ggml_cuda_info().devices[ctx.device].cc)) {
        return false;
    }
    // belt: never intercept a row-split weight (-sm row) — the raw data-pointer GEMV below
    // reads src0 as one dense local matrix. Production layouts are -sm layer (non-split).
    if (src0->buffer && ggml_backend_buffer_is_cuda_split(src0->buffer)) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }
    if (ggml_is_transposed(src0) || ggml_is_transposed(src1)) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    if (src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        if (src1->ne[1] != 1) { g_pxa_rf_declined_ne11++; pxa_rf_dbg("decline"); }
        return false;
    }
    // router shape family: many small output rows (experts), guard against ever matching a real
    // dense/output-head projection (those have ne01 in the tens-of-thousands / vocab-sized range,
    // or ne01==1 which mul_mat_1row above already owns).
    if (src0->ne[1] < 2 || src0->ne[1] > 4096) {
        return false;
    }
    const int ne00 = (int) src0->ne[0];
    const int ne01 = (int) src0->ne[1];
    if (pxa_router_fuse_mode_resolve() == 3) {
        // float4 loads. ne00 % 4 sizes the loop and nb[1] % 16 makes the ROW STRIDE safe,
        // but neither says anything about where the tensor actually STARTS, and the kernel
        // casts both src0->data and src1->data to float4*. A pool suballocation is not
        // required to be 16-byte aligned; when it is not, the first load is a misaligned-
        // address fault, i.e. an abort rather than a wrong number. Check the base pointers.
        if (ne00 % 4 != 0 || (src0->nb[1] % 16) != 0) return false;
        if ((((uintptr_t) src0->data) % 16) != 0 || (((uintptr_t) src1->data) % 16) != 0) {
            return false;
        }
        pxa_rf_fired_banner(3, ne00, ne01);
        k_pxa_router_gemv_f32_v2<<<ne01, 128, 0, ctx.stream()>>>(
            (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
            ne00, src0->nb[1]);
        return true;
    }
    pxa_rf_fired_banner(pxa_router_fuse_mode_resolve(), ne00, ne01);
    k_pxa_router_gemv_f32<<<ne01, WARP_SIZE, 0, ctx.stream()>>>(
        (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
        ne00, src0->nb[1]);
    return true;
}


// forward decl: definition lives with the other PXQ drivers (needs pxa_pxq4_bufs_on_device etc.)
static int pxa_pxq_mmv_2d(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                          const ggml_tensor * src1, ggml_tensor * dst);
static int pxa_pxq_gemm_2d(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                           const ggml_tensor * src1, ggml_tensor * dst);

static int ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cgraph * cgraph, int node_n) {

    if (ggml_cuda_router_gemv_f32(ctx, src0, src1, dst)) {
        return node_n;
    }

    // Tiny-R f32 GEMV, ny-capable (the hc_inject class) — see k_pxa_gemv_f32_ny.
    if (ggml_cuda_small_gemv_f32_ny(ctx, src0, src1, dst)) {
        return node_n;
    }

    // Small-R F16 decode GEMV (rev-2 per-head attn_gate class) — see k_pxa_gemv_f16.
    if (ggml_cuda_small_gemv_f16(ctx, src0, src1, dst)) {
        return node_n;
    }

    // Small-K / large-R F16 decode GEMV (hyper-connection up-projection class) -- the
    // transpose of the family above, which the R <= 512 cap there declines. See
    // k_pxa_gemv_f16_wide; the two predicates are disjoint, that cap is untouched.
    if (ggml_cuda_wide_gemv_f16(ctx, src0, src1, dst)) {
        return node_n;
    }

    // PXQ 2D decode mmv: plain MUL_MAT on a PXQ slab weight (shared experts, attention,
    // ssm_out, token_embd). Declines (-1) fall through to the stock dequant->cuBLAS path.
    if (pxa_pxq_mmv_2d(ctx, src0, src1, dst) == 0) {
        return node_n;
    }

    // PXQ 2D prefill GEMM: the same E==1 idiom for the ny > PXA_PXQ4_2D_MAX_NY window, so on a
    // gated arch a PXQ 2D weight never reaches the dequant->cuBLAS fallback. Default OFF.
    if (pxa_pxq_gemm_2d(ctx, src0, src1, dst) == 0) {
        return node_n;
    }

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;

    bool use_dequantize_mul_mat_vec = ggml_cuda_dmmv_type_supported(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src0->ne[0] % (GGML_CUDA_DMMV_X*2) == 0 && src1->ne[1] == 1;
    bool          use_mul_mat_vec_q =  ggml_is_quantized(src0->type) && !bad_padding_clear
        && ggml_cuda_mmvq_type_supported(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
    bool              use_mul_mat_q =  ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    // PXA_ADAPTIVE_DMMV (2026-06-15): per-CARD kernel selection, replacing the blunt compile-time
    // GGML_CUDA_FORCE_DMMV. Pascal (P100 sm_60 / 1080Ti sm_61) keeps the DMMV F16 fast path; Volta+
    // (V100 sm_70, tensor cores) prefers MMVQ. Decided per DEVICE by cc, so a mixed P100+V100 -sm-layer
    // rig automatically uses the best matmul kernel on EACH card in ONE binary (no -DGGML_CUDA_FORCE_DMMV).
    const int cc            = ggml_cuda_info().devices[ctx.device].cc;
    if (cc >= CC_VOLTA) {
        // mmvq (DP4A / tensor-core path) beats dmmv on Volta+; on Pascal we keep dmmv (no fast int8).
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && !use_mul_mat_vec_q;
    }

    bool any_gpus_with_slow_fp16 = false;
    use_mul_mat_q           = use_mul_mat_q           && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1]);
    any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16 || !fast_fp16_available(cc);

    // PXA_PASCAL_DMMV (2026-07-19, env-gated): the PXA_ADAPTIVE_DMMV intent (Pascal keeps the
    // DMMV F16 fast path) was dead code — this early return routed EVERY mmvq-capable quantized
    // GEMV (incl. the whole dense backbone at decode) to int8 MMVQ on all cards, and sm_60 has no
    // DP4A. With PXA_PASCAL_DMMV=1, a cc<CC_VOLTA device with a dmmv-supported type falls through
    // to the dequantize_mul_mat_vec branch below for ne11==1 GEMVs (loses the mmvq bias/TG fusion
    // for those nodes; A/B decides).
    // PXA_SPEC_SMALLN (B4 SPEC_VERIFY_ENGINE, 2026-07-22): on pre-Volta cards route the dense
    // quantized backbone at spec-verify batch sizes (ne11 2..8) to the multi-column dequant-FMA
    // GEMV instead of emulated-dp4a MMVQ. See pxa-smalln.cuh. Default OFF (PXA_SPEC_SMALLN=1).
    static const bool pxa_spec_smalln = getenv("PXA_SPEC_SMALLN") && atoi(getenv("PXA_SPEC_SMALLN")) != 0;
    // PXA_SMALLN_GEMV (2026-08-03): route the ne11==1 dense backbone GEMV through the same
    // dequant-FMA kernel on sm_60. Profiled (nsys, 122B PXQU48, 4xP100, deep fill): the mxfp4
    // dense matvecs on emulated-dp4a MMVQ were 14.2 ms/token (35% of decode wall), q8_0 (incl.
    // the 248320-row head) another 3.0 ms. The legacy dmmv template was A/B'd first and LOST
    // (24.97 -> 14.7 t/s deep decode: divergent global-table lookups + per-pair scale decode);
    // k_pxa_smalln_* stages the table in smem and decodes the scale once per block.
    // MEASURED (2026-08-03, 122B PXQU48 4xP100, banner-verified): the R=1 route is a LOSS too —
    // deep decode 25.4 -> 18.4 t/s (b2048), 26.1 -> 19.0 (b512). Better than dmmv's 14.7 but
    // still -28% vs emulated-dp4a MMVQ. Verdict: the sm_60 dense GEMV is INSTRUCTION-economy
    // bound, not bandwidth bound — mmvq's packed-q8_1 integer form beats scalar dequant-FMA
    // (2 loads + table + 2 FMA per weight pair) even with dp4a emulated. Default OFF everywhere;
    // env-only for interleaved A/Bs. A winning replacement must cut instructions/weight
    // (half2 __hfma2 math, packed nibble decode), not just avoid the int8 path.
    static const int pxa_smalln_gemv = [](){
        const char * e = getenv("PXA_SMALLN_GEMV");
        return e ? atoi(e) : 0;
    }();
    const bool pxa_smalln_take_gemv = pxa_smalln_gemv == 1 && src1->ne[1] == 1;
    if ((pxa_smalln_take_gemv || (pxa_spec_smalln && src1->ne[1] >= 2 && src1->ne[1] <= 8))
        && cc < CC_VOLTA && ggml_cuda_pxa_smalln_supported(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[2]*src1->ne[3] == 1 && src0->ne[2] == 1 && src0->ne[3] == 1
        && src0->ne[0] % 32 == 0 && !bad_padding_clear
        && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        if (pxa_smalln_take_gemv) {
            static std::atomic<bool> logged{false};
            bool exp = false;
            if (logged.compare_exchange_strong(exp, true)) {
                fprintf(stderr, "PXA_SMALLN_GEMV: FIRING (cc=%d, %s ne11=1 GEMV -> smalln dequant-FMA; PXA_SMALLN_GEMV=0 reverts)\n",
                        cc, ggml_type_name(src0->type));
            }
        }
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_pxa_smalln, nullptr);
        return node_n;
    }

    // PXA_PASCAL_DMMV stays ENV-ONLY (default OFF) — the sm_60 auto-arm was A/B'd 2026-08-03 on
    // the 122B PXQU48 4xP100 rig and MEASURED A LOSS: deep decode 24.97 -> 14.7 t/s (-41%) with
    // mxfp4+q8_0 dense GEMVs on the legacy dmmv template (divergent device-global kvalues
    // lookups, per-pair E8M0 decode, 64-thread low-ILP blocks). The sm_60 GEMV fix that WON is
    // candidate is PXA_SMALLN_GEMV above (its own A/B pending). Kept env-gated for experiments.
    static const bool pxa_pascal_dmmv = getenv("PXA_PASCAL_DMMV") && atoi(getenv("PXA_PASCAL_DMMV")) != 0;
    const bool pxa_dmmv_take = pxa_pascal_dmmv && cc < CC_VOLTA && use_dequantize_mul_mat_vec;
    if ((use_mul_mat_vec_q || use_mul_mat_q) && src1->ne[2]*src1->ne[3] == 1 && !pxa_dmmv_take) {
        return ggml_cuda_mul_mat_q(ctx, src0, src1, dst, cgraph, node_n, use_mul_mat_vec_q);
    }

    // PXA_SPEC_1ROW (default ON, =0 rollback to the ne11==1-only dispatch): single-output-row
    // GEMV, now also at spec-verify batch sizes (Ny<=8, the MMVQ_MAX_BATCH_SIZE convention).
    // Layout/type belts mirror the router-fuse checks above; shapes that fail them fall through
    // to the generic paths exactly as before. Note this also retires the old GGML_ABORT for a
    // 1-row quantized src0 that missed mmvq/mmq — those now fall through to the cuBLAS dequant
    // path instead of aborting.
    static const bool pxa_spec_1row = !(getenv("PXA_SPEC_1ROW") && atoi(getenv("PXA_SPEC_1ROW")) == 0);
    if (ggml_nrows(src0) == 1 && (src1->ne[1] == 1 || (pxa_spec_1row && src1->ne[1] <= 8)) && src1->ne[2]*src1->ne[3] == 1
        && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && ggml_is_contiguous(src0) && src1->nb[0] == sizeof(float)
        && !(src0->buffer && ggml_backend_buffer_is_cuda_split(src0->buffer))) {
        // PXA_SHEXP_GATE_FUSE: defer the shared-expert 1-row gate GEMV into its
        // fused_mul_sigmoid consumer (one launch instead of two, bit-exact; kernel + gates
        // above mul_mat_1row). Any pattern mismatch falls through to mul_mat_1row unchanged.
        if (pxa_shexp_gate_try_defer(ctx, src0, src1, dst, cgraph, node_n)) {
            return node_n;
        }
        mul_mat_1row(src0, src1, dst, ctx);
        return node_n;
    }
    bool debug = false; //src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F32;

    // debug helpers
    //printf("src0: %8d %8d %8d %8d\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    //printf("src1: %8d %8d %8d %8d\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    //printf("src0 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
    //printf("src1 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);

    if (any_gpus_with_slow_fp16 && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        if (debug) printf("%s(%s): using ggml_cuda_mul_mat_vec_p021\n", __func__, dst->name);
        // FP32 precision KQ single-batch for batch size 1 without FlashAttention
        ggml_cuda_mul_mat_vec_p021(ctx, src0, src1, dst);
    } else if (any_gpus_with_slow_fp16 && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1) {
        if (debug) printf("%s(%s): using ggml_cuda_mul_mat_vec_nc\n", __func__, dst->name);
        // FP32 precision KQV single-batch for batch size 1 without FlashAttention
        ggml_cuda_mul_mat_vec_nc(ctx, src0, src1, dst);
    } else if ((src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32) && (src1->type == src0->type || !any_gpus_with_slow_fp16)
               && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2]*src1->ne[3] > 1) {
        if (debug) printf("%s(%s): ggml_cuda_mul_mat_batched_cublas\n", __func__, dst->name);
        // KQ + KQV multi-batch without FlashAttention
        ggml_cuda_mul_mat_batched_cublas(ctx, src0, src1, dst);
    } else if (use_dequantize_mul_mat_vec) {
        if (debug) printf("%s(%s): ggml_cuda_op_mul_mat(ggml_cuda_op_dequantize_mul_mat_vec)\n", __func__, dst->name);
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_dequantize_mul_mat_vec, nullptr);
    } else if (use_mul_mat_vec_q) {
        if (debug) printf("%s(%s): ggml_cuda_op_mul_mat(ggml_cuda_op_mul_mat_vec_q)\n", __func__, dst->name);
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_vec_q, quantize_row_q8_1_cuda);
    } else if (use_mul_mat_q) {
        if (debug) printf("%s(%s): ggml_cuda_op_mul_mat(ggml_cuda_op_mul_mat_q)\n", __func__, dst->name);
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_q, quantize_mmq_q8_1_cuda);
    } else {
        if (debug) printf("%s(%s, %s): ggml_cuda_op_mul_mat(ggml_cuda_op_mul_mat_cublas)\n", __func__, dst->name, ggml_type_name(src0->type));
        // PXA_GEMV_DBG=1: name every decode-time GEMV that lands on the cuBLAS fallback, once
        // per tensor name. Profiling found 48 anonymous f32 GEMVs/token at 25.7us each here.
        static const bool pxa_gemv_dbg = getenv("PXA_GEMV_DBG") && atoi(getenv("PXA_GEMV_DBG")) != 0;
        if (pxa_gemv_dbg && src1->ne[1] == 1) {
            static std::mutex mtx; static std::set<std::string> seen;
            std::lock_guard<std::mutex> lk(mtx);
            if (seen.insert(src0->name).second) {
                fprintf(stderr, "PXA_GEMV_DBG: cublas GEMV src0=%s type=%s ne=[%ld,%ld] src1_ne1=%ld\n",
                        src0->name, ggml_type_name(src0->type), (long)src0->ne[0], (long)src0->ne[1], (long)src1->ne[1]);
            }
        }
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_cublas, nullptr);
    }
    return node_n;
}

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

template <typename data_t = float>
static __global__ void k_copy_src_to_contiguous(const char * __restrict__ src_original, char * __restrict__ src_contiguous,
                                                  const mmid_row_mapping * __restrict__ row_mapping,
                                                  int64_t ne10, int64_t ne11, size_t nb11, size_t nb12) {
    int32_t i = blockIdx.x;

    const int32_t i11 = row_mapping[i].i1 % ne11;
    const int32_t i12 = row_mapping[i].i2;

    data_t * src_row_contiguous = (data_t *)(src_contiguous + i*nb11);
    const data_t * src_row_original = (const data_t *)(src_original + i11*nb11 + i12*nb12);

    for (int j = threadIdx.x; j < ne10; j += blockDim.x) {
        src_row_contiguous[j] = src_row_original[j];
    }
}

static __global__ void k_copy_dst_from_contiguous(char * __restrict__ dst_original, const char * __restrict__ dst_contiguous,
                                                  const mmid_row_mapping * __restrict__ row_mapping,
                                                  int64_t ne0,
                                                  size_t nb1, size_t nb2) {
    int32_t i = blockIdx.x;

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;
    if (i1 < 0) return;   // device-built map: slot past the routed total, nothing to scatter

    const float * dst_row_contiguous = (const float *)(dst_contiguous + i*nb1);
    float * dst_row_original = (float *)(dst_original + i1*nb1 + i2*nb2);

    for (int j = threadIdx.x; j < ne0; j += blockDim.x) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

//static __global__ void k_quick_add(uint32_t n, uint32_t n_per_row, const float * src1, const float * src2, float * dst) {
//
//    for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
//        dst[j] = src1[j] + src2[j % n_per_row];
//    }
//}

static __global__ void k_quick_add(uint32_t n_per_row, const float * src1, const float * src2, float * dst) {

    uint32_t row = blockIdx.x;
    const float * src1_row = src1 + row*n_per_row;
    float * dst_row = dst + row*n_per_row;

    for (uint32_t j = threadIdx.x; j < n_per_row; j += blockDim.x) {
        dst_row[j] = src1_row[j] + src2[j];
    }
}

// PXA_MOE_DEVICE_MAP: 0 = the host row mapping below; 1 = the device kernels in
// ggml-cuda/pxq-moemap.cuh; 2 = those kernels plus a host cross-check of the buffers they build.
// Default: pxa_moe_device_map_default() -- mode 1 at ENHANCE on an sm_60 topology (the 4x P100
// seat this was measured on), 0 elsewhere. Explicit env wins in both directions.
static inline int pxa_moe_device_map_mode() {
    static const int v = [] {
        const char * e = getenv("PXA_MOE_DEVICE_MAP");
        return e ? atoi(e) : pxa_moe_device_map_default();
    }();
    return v;
}
static inline bool pxa_moe_device_map() { return pxa_moe_device_map_mode() != 0; }

static inline bool prepare_row_mappigs(ggml_backend_cuda_context& ctx, int64_t n_as, int64_t n_ids,
        const ggml_tensor * ids, std::vector<int>& moe_counts, std::vector<int>& cum_moe_counts,
        ggml_cuda_pool_alloc<mmid_row_mapping>& dev_row_mapping) {

    GGML_ASSERT(moe_counts.empty() && cum_moe_counts.empty());

    // With PXA_MOE_DEVICE_MAP on, every remaining visit here is a D2H + cudaStreamSynchronize
    // still sitting inside graph compute. Name the first few so the un-migrated call sites are
    // identifiable from a server log instead of from a profiler.
    if (pxa_moe_device_map()) {
        static std::atomic<int> left{16};
        if (left.fetch_sub(1) > 0) {
            fprintf(stderr, "PXA_MOE_HOST_MAP dev%d: ids=%s n_as=%d n_ids=%d rows=%d (host readback + sync)\n",
                    ctx.device, ids->name, (int)n_as, (int)n_ids, (int)(ids->ne[1]*n_ids));
        }
    }

    auto stream = ctx.stream();

    std::vector<char> ids_host(ggml_nbytes(ids));
    const char * ids_dev = (const char *) ids->data;
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids_dev, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<mmid_row_mapping> rmapping(ids->ne[1]*n_ids);
    moe_counts.resize(n_as, 0);
    cum_moe_counts.resize(n_as + 1);

    bool is_ser = false;
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
            if (row_id_i >= 0 && row_id_i < n_as) ++moe_counts[row_id_i];
            else is_ser = true;
        }
    }
    cum_moe_counts[0] = 0;
    for (int i = 0; i < (int)n_as; ++i) {
        cum_moe_counts[i+1] = cum_moe_counts[i] + moe_counts[i];
    }

    dev_row_mapping.alloc(cum_moe_counts[n_as]);

    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
            if (row_id_i >= 0 && row_id_i < n_as) {
                rmapping[cum_moe_counts[row_id_i]++] = {(int)id, (int)iid1};
            }
        }
    }

    for (int i = 0; i < (int)n_as; ++i) cum_moe_counts[i] -= moe_counts[i];

    CUDA_CHECK(cudaMemcpyAsync(dev_row_mapping.get(), rmapping.data(),
                cum_moe_counts[n_as]*sizeof(mmid_row_mapping), cudaMemcpyHostToDevice, stream));
    //CUDA_CHECK(cudaStreamSynchronize(stream));

    return is_ser;
}

static bool ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * next) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];


    CUDA_CHECK(cudaMemsetAsync((char *)dst->data, 0, ggml_nbytes(dst), ctx.stream()));

    if (src1->ne[1] <= MMVQ_MAX_BATCH_SIZE && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_quantized(src0->type) &&
        ggml_cuda_mmvq_type_supported(src0->type) &&   // no mmvq kernel (e.g. PXQ slabs) -> per-expert dequant/cublas loop below
        !pxa_pxq_mmvq_type(src0->type) &&              // PXQ MMVQ is dense-decode only; MoE keeps its grouped drivers
        ggml_backend_buffer_is_cuda(src0->buffer) &&
        ggml_backend_buffer_is_cuda(src1->buffer) &&
        ggml_backend_buffer_is_cuda(dst->buffer) &&
        src1->type == GGML_TYPE_F32) {
        int device_id = ctx.device;
        ggml_backend_cuda_buffer_context * src0_ctx = (ggml_backend_cuda_buffer_context *) src0->buffer->context;
        ggml_backend_cuda_buffer_context * src1_ctx = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx  = (ggml_backend_cuda_buffer_context *) dst->buffer->context;
        if (src0_ctx->device == device_id &&
            src1_ctx->device == device_id &&
            dst_ctx->device  == device_id) {
            GGML_ASSERT(src1->ne[0] % QK8_1 == 0);
            // Fast TG path
            const int64_t n_ids = ids->ne[0];
            auto stream = ctx.stream(device_id, 0);

            auto local_dst = *dst;
            local_dst.ne[2] = n_ids;
            local_dst.ne[1] = local_dst.ne[3] = 1;
            local_dst.nb[2] = local_dst.nb[1];

            const int64_t src1_padded_col_size = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
            auto src_1_ddq_size = src1_padded_col_size*sizeof(block_q8_1)/QK8_1;
            auto local_src1 = *src1;
            local_src1.ne[1] = 1;
            local_src1.nb[1] = src_1_ddq_size;
            local_src1.nb[2] = src1->ne[1] > 1 ? src_1_ddq_size : 0;
            local_src1.nb[3] = local_src1.nb[2];

            ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool());
            local_src1.data = src1_quantized.alloc(src_1_ddq_size*src1->ne[1]);
            quantize_row_q8_1_cuda((const float *)src1->data, (void *)src1_quantized.get(), src1->ne[0], src1->ne[1], 1,
                    src1_padded_col_size, src0->type, stream);
            CUDA_CHECK(cudaGetLastError());

            ggml_cuda_op_mul_mat_vec_q_id(ctx, src0, &local_src1, ids, &local_dst, nullptr,
                (const char *)src0->data, nullptr, src1_quantized.get(), (float *)dst->data,
                0, src0->ne[1], 1, src1_padded_col_size, stream);
            CUDA_CHECK(cudaGetLastError());

            if (next && next->op == GGML_OP_MUL_MAT_ID && next->src[0]->type == src0->type && src1 == next->src[1] &&
                ggml_are_same_shape(src0, next->src[0]) &&
                ggml_backend_buffer_is_cuda(next->src[0]->buffer) &&
                ggml_backend_buffer_is_cuda(next->buffer)) {
                ggml_backend_cuda_buffer_context * next_src0_ctx = (ggml_backend_cuda_buffer_context *) next->src[0]->buffer->context;
                ggml_backend_cuda_buffer_context * next_dst_ctx  = (ggml_backend_cuda_buffer_context *) next->buffer->context;
                if (next_src0_ctx->device == device_id &&
                    next_dst_ctx->device  == device_id) {
                    local_dst.data = next->data;
                    ggml_cuda_op_mul_mat_vec_q_id(ctx, next->src[0], &local_src1, ids, &local_dst, nullptr,
                        (const char *)next->src[0]->data, nullptr, src1_quantized.get(), (float *)next->data,
                        0, src0->ne[1], 1, src1_padded_col_size, stream);
                    CUDA_CHECK(cudaGetLastError());
                    return true;
                }
            }

            return false;
        }
    }

    if (src1->ne[2] <= ctx.mmq_id_thresh*src0->ne[2] &&
        ggml_is_quantized(src0->type) && ggml_cuda_can_use_mmq_id(src0->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[2])) {
        ggml_cuda_mul_mat_q_id(ctx, src0, src1, ids, dst, nullptr, nullptr);
        return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row  = *dst;

    char * src0_original = (char *) src0->data;
    char * src1_original = (char *) src1->data;
    char * dst_original  = (char *)  dst->data;

    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src0_row.nb[3] = nb02;

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;


    ggml_cuda_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool());
    std::vector<int> moe_counts, cum_moe_counts;
    bool is_ser = prepare_row_mappigs(ctx, n_as, n_ids, ids, moe_counts, cum_moe_counts, dev_row_mapping);
    if (is_ser) {
        CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
    }

    ggml_cuda_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(src1));
    ggml_cuda_pool_alloc<char>  dst_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));

    src1_row.data = src1_contiguous.get();
    dst_row.data  =  dst_contiguous.get();

    for (int64_t i02 = 0; i02 < n_as; i02++) {

        int64_t num_src1_rows = moe_counts[i02];

        if (num_src1_rows == 0) {
            continue;
        }

        size_t mapping_offset = cum_moe_counts[i02];

        {
            dim3 block_dims(std::min((unsigned int)ne10, 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_src_to_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    src1_original, src1_contiguous.get(), dev_row_mapping.get() + mapping_offset, ne10, ne11, nb11, nb12);
            CUDA_CHECK(cudaGetLastError());
        }

        src0_row.data = src0_original + i02*nb02;

        GGML_ASSERT(nb11 == sizeof(float)*ne10);
        GGML_ASSERT(nb1 == sizeof(float)*ne0);

        src1_row.ne[1] = num_src1_rows;
        src1_row.nb[1] = nb11;
        src1_row.nb[2] = num_src1_rows*nb11;
        src1_row.nb[3] = num_src1_rows*nb11;

        dst_row.ne[1] = num_src1_rows;
        dst_row.nb[1] = nb1;
        dst_row.nb[2] = num_src1_rows*nb1;
        dst_row.nb[3] = num_src1_rows*nb1;

        if (ggml_is_quantized(src0->type) &&
            ggml_cuda_should_use_mmq(src0->type, ggml_cuda_info().devices[ctx.device].cc, num_src1_rows)) {
            auto src1_padded_num_cols = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
            auto src1_padded_row_size = src1_padded_num_cols/ggml_blck_size(GGML_TYPE_Q8_1)*ggml_type_size(GGML_TYPE_Q8_1);
            auto src1_quantized_size  = src1_padded_row_size*num_src1_rows;
            if (true || num_src1_rows > MMVQ_MAX_BATCH_SIZE) {
                src1_quantized_size += get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
                ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), src1_quantized_size);
                quantize_mmq_q8_1_cuda((const float *)src1_contiguous.get(), src1_quantized.get(), ne00, num_src1_rows, 1,
                        src1_padded_num_cols, src0->type, stream);
                src1_row.nb[1] = src1_padded_row_size;
                src1_row.nb[2] = src1_row.nb[3] = src1_row.nb[1]*num_src1_rows;
                ggml_cuda_mul_mat_q_id(ctx, &src0_row, &src1_row, nullptr, &dst_row, nullptr, src1_quantized.get());

                CUDA_CHECK(cudaGetLastError());
            } else {
                ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), src1_quantized_size);
                quantize_row_q8_1_cuda((const float *)src1_contiguous.get(), src1_quantized.get(), ne00, num_src1_rows, 1,
                        src1_padded_num_cols, src0->type, stream);
                src1_row.nb[1] = src1_padded_row_size;
                src1_row.nb[2] = src1_row.nb[3] = src1_row.nb[1]*num_src1_rows;
                ggml_cuda_op_mul_mat_vec_q(ctx, &src0_row, &src1_row, &dst_row, (const char *)src0_row.data, nullptr,
                        src1_quantized.get(), (float *)dst_row.data,
                        0, src0_row.ne[1], num_src1_rows, src1_padded_num_cols, stream);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            ggml_cuda_mul_mat(ctx, &src0_row, &src1_row, &dst_row, nullptr, 0);
        }

        {
            dim3 block_dims(std::min((unsigned int)ne0, 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_dst_from_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    dst_original, dst_contiguous.get(),
                    dev_row_mapping.get() + mapping_offset,
                    ne0,
                    nb1, nb2);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    return false;
}

#include "ggml-cuda/grouped_moe_verify.cuh"
#include "ggml-cuda/pxa/pxq4.cuh"
#include "ggml-cuda/pxa/pxq-moemap.cuh"
#include "ggml-cuda/pxa/pxq6.cuh"
#include "ggml-cuda/pxa/pxq6i8.cuh"
#include "ggml-cuda/pxa/pxq4-v70.cuh"
#include "ggml-cuda/pxa/pxq4-mmq.cuh"

// ============================== PXQ4 (PXA-native quant) drivers ==============================
// PXQ4 = MXFP4 numerics in a GEMM-tile-ordered slab layout (see pxq4.cuh). These drivers wire the
// fused kernels into the MoE dispatch: prefill = ONE grouped launch over all routed experts
// (replaces the per-expert dequant+cublas loop that starved the P100), decode = fused
// up+gate+swiglu / down mmv straight from f32 activations (no q8_1 stage, no host syncs).
// Both return -1 to decline (shape/type/arch not handled) -> the stock fallback paths run,
// which for PXQ4 means dequant(convert.cu)->cublas: functionally correct on every arch.

static bool pxa_pxq4_bufs_on_device(ggml_backend_cuda_context & ctx, std::initializer_list<const ggml_tensor *> ts) {
    for (const ggml_tensor * t : ts) {
        if (!t || !t->buffer || !ggml_backend_buffer_is_cuda(t->buffer)) return false;
        if (((ggml_backend_cuda_buffer_context *) t->buffer->context)->device != ctx.device) return false;
    }
    return true;
}

// ============================ PXQ 2D decode mmv (non-MoE MUL_MAT) ============================
// The fused PXQ drivers above are all wired into the MoE dispatch (MUL_MAT_ID over routed expert
// stacks). A plain 2D MUL_MAT on a PXQ weight -- shared experts (ffn_*_shexp), attention
// projections, ssm_out, token_embd, nextn.eh_proj -- found no fast path and fell through to
// dequant(convert.cu)->cuBLAS, which re-expands the whole tensor to f16 in GLOBAL memory. Prefill
// amortizes that over the batch; decode pays it per token. Measured cost of that fallback when the
// shared experts were converted to PXQ4: decode 105.4 -> 63.5 t/s at 66-token fill (-39.7%) with
// prefill unchanged (-0.7%) -- the exact signature of a per-call dequant.
//
// This driver closes that hole WITHOUT new kernel math: a 2D weight is just the E==1 case of the
// expert-stacked mmv. We reuse k_pxq6_mmv verbatim (same template instantiations, same dot code)
// and feed it a one-entry ids buffer holding 0 with BOTH id strides zeroed, so every block reads
// e == 0 and pxq6_panel() degenerates to W + p*stride. The slot (j) axis collapses to a single
// plane via zero x/dst slot strides. Numerics are therefore identical to the proven MoE path by
// construction, not by re-derivation.
//
// Gate: PXA_PXQ4_2D=1 on, =0 off. PXA_PXQ4_2D_MAX_NY caps the token count we claim (default 8):
// beyond that the dequant->cuBLAS GEMM amortizes and is the better path, so we decline.
//
// DEFAULT ON since BACKBONE_REV 2 (2026-07-26). Until then no shipped file carried a PXQ 2D
// weight, so the default was irrelevant; the rev-2 backbone table puts attention, shared
// experts and dense FFN on native PXQ types in EVERY tier, which makes this driver the
// difference between a fused decode and a per-token dequant->cuBLAS expansion of the whole
// tensor (measured -39.7% decode, 105.4 -> 63.5 t/s, when the shared experts alone went PXQ4).
// Without the flip the recipe fix would be blamed on the codec. PXA_PXQ4_2D=0 restores the
// old default for the interleaved A/B.
static bool pxa_pxq4_2d_enabled() {
    static const bool v = [] {
        const char * e = getenv("PXA_PXQ4_2D");
        return !(e && atoi(e) == 0);
    }();
    return v;
}

static int pxa_pxq4_2d_max_ny() {
    static const int v = [] {
        const char * e = getenv("PXA_PXQ4_2D_MAX_NY");
        return e ? atoi(e) : 8;
    }();
    return v;
}

// K1-2D KSPLIT: re-grid the four in-block k-segment chains of k_pxq6_mmv into four 64-thread
// blocks + a fixed-order workspace reducer, so a launch that cannot put a block on every SM
// spreads over 4x more of them. BIT-IDENTICAL to the single-launch form (same per-thread fp32
// chains, same ascending 4-term reduction) -- verified: temp-0 sha256 unchanged vs the pre-split
// binary on every arm incl. always-split. Default OFF: measured FLAT on the 2xV100 attn-native
// config (panels 32..128 of 80 SMs), so it buys nothing there and costs one extra launch per
// firing node. Kept for the shapes the occupancy model still favours (shexp-native panels=8,
// low-SM cards) and for the rigorous interleaved A/B. PXA_PXQ4_2D_KSPLIT=1 enables.
static bool pxa_pxq4_2d_ksplit() {
    static const bool v = [] {
        const char * e = getenv("PXA_PXQ4_2D_KSPLIT");
        return e && atoi(e) != 0;
    }();
    return v;
}

// K8-2D S-SPLIT gate (2026-07-26). Default ON; PXA_PXQ4_2D_SPLIT=0 rolls the dispatch back to
// the unsplit/K1 forms. Motivation (nvprof gpu-trace, Laguna-XS rev-2, 2xP100): the unsplit
// grid is panels = R/64 blocks, so R=2048 K=8192 (attn_output/ffn_down) = 32 blocks whose
// 33.8 KB x-stage caps occupancy at 1 block/SM -> 87 GB/s, and R=512 (shexp up/gate) = 8
// blocks -> 20 GB/s, while the SAME dot path with a healthy grid (R=8192: 128 blocks, 9 KB)
// runs 210 GB/s and BEATS mmvq-MXFP4 (161 GB/s) per byte. The S-split multiplies blocks by S
// and divides the x-stage by S, attacking both. NOT bit-exact vs the unsplit kernel (chunk
// summation order) -- deterministic run-to-run; G3/G4 gated like K1b/K6.
static bool pxa_pxq4_2d_split() {
    static const bool v = [] {
        const char * e = getenv("PXA_PXQ4_2D_SPLIT");
        return !(e && atoi(e) == 0);
    }();
    return v;
}

// Grid-coverage target the split aims for (blocks incl. the ny axis). Default 2x the device's
// SM count; PXA_PXQ4_2D_SPLIT_TARGET overrides (0 = never split; large = always max split).
// TUNED 2026-07-27 (Qwable-27B dense, decode, 5 reps/arm, spread < 0.4%). The shipped 2*nsm
// default sat far below the knee on both target Teslas -- it only ever split the shapes whose
// x-stage forced it, leaving every well-shaped node at 1 block/SM:
//   2xP100 (nsm 56): T=112 (2*nsm) 14.81 t/s | 224 17.45 | 336 19.17 | 448 19.17 | 560 19.15
//                    | 672 18.86 | 1120 18.11 | 2240 16.44   -> knee 336-560, take 8*nsm = 448
//   2xV100 (nsm 80): T=160 (2*nsm) 27.89 t/s | 320 29.31 | 640 29.59 | 1280 30.70 | 2560 30.19
//                    -> knee 1280 = 16*nsm
// Both are broad plateaus, not spikes. Other arches keep 2*nsm (unmeasured -- do not guess).
static int pxa_pxq4_2d_split_target(int device) {
    static const int ov = [] {
        const char * e = getenv("PXA_PXQ4_2D_SPLIT_TARGET");
        return e ? atoi(e) : -1;
    }();
    if (ov >= 0) return ov;
    const int nsm = ggml_cuda_info().devices[device].nsm;
    const int cc  = ggml_cuda_info().devices[device].cc;
    if (cc == 700) return 16*nsm;
    if (cc == 600) return  8*nsm;
    return 2*nsm;
}

// First firing / first decline per device (same phantom-lever discipline as the K1 log).
static void pxa_pxq4_2d_split_log(int device, bool fired, int S, int panels, int ny, int R, int K) {
    static std::atomic<uint32_t> seen{0};
    if (device < 0 || device >= 16) return;
    const uint32_t bit = 1u << (2*device + (fired ? 0 : 1));
    if (seen.fetch_or(bit) & bit) return;
    fprintf(stderr, "PXA_PXQ4_2D_SPLIT dev%d: %s (S=%d panels=%d ny=%d R=%d K=%d, target %d blocks)\n",
            device, fired ? "FIRING" : "DECLINED -> unsplit launch", S, panels, ny, R, K,
            pxa_pxq4_2d_split_target(device));
}

// MoE down split (2026-08-04). The decode fused down (the MUL_MAT_ID right after the gateup in
// pxa_pxq4_moe_fast_tg) launches k_pxq6_mmv on panels*n_ids blocks -- 48*8 = 384 on the 122B
// down (R=3072 K=1024 n_ids=8), ~1.14 waves at the fired variants' 6 blocks/SM on a 56-SM
// P100: one nearly-full wave plus a 14% tail, 120 GB/s achieved -- the worst weight kernel in
// the F1 profile, while every other big node is split to 2+ waves. This lever routes that node
// through the split kernels the driver already owns (both take real ids/n_as/n_ids; only the
// MUL_MAT_ID routing was missing):
//   PXA_PXQ_MOE_DOWN_SPLIT=1     -> K1 kseg split: k_pxq6_mmv_ksplit (64-thr,
//                                   panels*KSEG*n_ids blocks) + k_pxq_mmv_reduce. ws is
//                                   KSEG*R per slot -- on the down shape ~786 KB/launch of
//                                   round-trip (~11% of the node's bytes). Break-even needs
//                                   only ~133 GB/s achieved vs today's 120.
//   PXA_PXQ_MOE_DOWN_SPLIT=2/4/8 -> gen S-split: k_pxq6_mmv_ksplit_gen +
//                                   k_pxq_mmv_reduce_s. ws is NFIX*KSEG*R RAW canonical
//                                   partials per slot INDEPENDENT of S -- on the down shape
//                                   (R=3072, nfix=8) ~6.0 MB/launch of round-trip, ~81% of
//                                   the node's bytes: the wave-fill gain must beat that tax
//                                   (break-even ~216 GB/s total, above the gateup family's
//                                   measured 194). Measured before believed; kill on loss.
// Default 0 = OFF, byte-identical dispatch. Both arms BIT-EXACT vs the unsplit kernel by
// construction (PXQ_CANON_v1): k_pxq6_mmv, k_pxq6_mmv_ksplit, k_pxq6_mmv_ksplit_gen and
// k_pxq_mmv_reduce_s ALL fold with nfix = pxq6_canon_nfix(kslabs, PXQ6_MMV_SPLIT_MAX). That
// MMV cap (NOT the gateup family's PXQ6_GU_SPLIT_MAX) is load-bearing twice: at K >= 2048 the
// GU cap changes the canonical fold (canon_nfix(64,8)=8 != canon_nfix(64,16)=16 => silent
// bit-break), and the gen ws row stride is a fixed PXQ6_MMV_SPLIT_MAX*PXQ4_MMV_KSEG slots, so
// sizing the workspace with the GU cap would undersize it 2x => OOB writes.
static inline int pxa_pxq_moe_down_split() {
    static const int s = [](){
        const char * e = getenv("PXA_PXQ_MOE_DOWN_SPLIT");
        int v = e ? atoi(e) : 0;
        if (v != 0 && v != 1 && v != 2 && v != 4 && v != 8) {
            fprintf(stderr, "PXA: PXA_PXQ_MOE_DOWN_SPLIT=%d invalid (want 0/1/2/4/8) -- OFF\n", v);
            v = 0;
        }
        if (v == 1) fprintf(stderr, "PXA_PXQ_MOE_DOWN_SPLIT: K1 kseg split armed (MoE fused down; bit-exact)\n");
        if (v >= 2) fprintf(stderr, "PXA_PXQ_MOE_DOWN_SPLIT: gen S=%d armed (MoE fused down; bit-exact)\n", v);
        return v;
    }();
    return s;
}

// First firing / first decline per device (phantom-lever discipline, same as the K8-2D log):
// an armed lever that never prints FIRING did not engage -- the A/B is void, not null.
static void pxa_pxq_moe_down_split_log(int device, bool fired, int S, int gx, int n_ids, int ny,
                                       int R, int K, int nfix) {
    static std::atomic<uint32_t> seen{0};
    if (device < 0 || device >= 16) return;
    const uint32_t bit = 1u << (2*device + (fired ? 0 : 1));
    if (seen.fetch_or(bit) & bit) return;
    fprintf(stderr, "PXA_PXQ_MOE_DOWN_SPLIT dev%d: %s (S=%d grid=%dx%dx%d=%d blocks R=%d K=%d nfix=%d)\n",
            device, fired ? "FIRING" : "DECLINED -> unsplit fused down", S, gx, n_ids, ny,
            gx*n_ids*ny, R, K, nfix);
}

// Persistent one-entry {0} ids buffer per device: every PXQ 2D node feeds the expert-stacked
// kernels through the E==1 idiom, and a fresh pool alloc + cudaMemsetAsync per node was pure
// per-call tax (200+ decode calls/token on a rev-2 backbone). cudaMalloc is illegal mid-capture:
// callers get nullptr there and fall back to the pool form (replayed graphs stay correct).
static const int32_t * pxa_pxq_zero_ids(int device, cudaStream_t stream) {
    static int32_t * bufs[64] = {nullptr};
    static std::mutex mtx;
    if (device < 0 || device >= 64) return nullptr;
    if (bufs[device]) return bufs[device];
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &st);
    if (st != cudaStreamCaptureStatusNone) return nullptr;
    std::lock_guard<std::mutex> lk(mtx);
    if (!bufs[device]) {
        int32_t * p = nullptr;
        if (cudaMalloc(&p, sizeof(int32_t)) != cudaSuccess) { (void)cudaGetLastError(); return nullptr; }
        if (cudaMemset(p, 0, sizeof(int32_t)) != cudaSuccess) { (void)cudaGetLastError(); cudaFree(p); return nullptr; }
        bufs[device] = p;
    }
    return bufs[device];
}

// Fire the re-grid only below a block-count threshold; at or above it the plain grid already
// covers the machine and the extra launch is pure tax. Default = the device's SM count.
// PXA_PXQ4_2D_KSPLIT_MINBLK overrides (0 = never split, a huge value = always split) for env-only A/B.
static int pxa_pxq4_2d_ksplit_minblk(int device) {
    static const int ov = [] {
        const char * e = getenv("PXA_PXQ4_2D_KSPLIT_MINBLK");
        return e ? atoi(e) : -1;
    }();
    return ov >= 0 ? ov : ggml_cuda_info().devices[device].nsm;
}

// The re-grid is bit-exact, so a config where it silently declines (no instantiation, or the
// workspace can't grow under graph capture) is INDISTINGUISHABLE from one where it fires. Log the
// first firing and the first decline per device so nobody has to chase a phantom lever.
static void pxa_pxq4_2d_ksplit_log(int device, bool fired, int panels, int ny, int R, int K) {
    static std::atomic<uint32_t> seen{0};
    if (device < 0 || device >= 16) return;
    const uint32_t bit = 1u << (2*device + (fired ? 0 : 1));
    if (seen.fetch_or(bit) & bit) return;
    fprintf(stderr, "PXA_PXQ4_2D_KSPLIT dev%d: %s (panels=%d ny=%d R=%d K=%d, split below %d blocks)\n",
            device, fired ? "FIRING" : "DECLINED -> unsplit launch", panels, ny, R, K,
            pxa_pxq4_2d_ksplit_minblk(device));
}


// PXA_PXQ_DISPATCH_DBG=1 -> print each distinct (up,gate) format pair the fused MoE drivers
// actually accept, once. Exists so "the new tier is on the fused path" is a PRINTED FACT rather
// than an inference from a speed delta (a decline is silent and looks exactly like a slow kernel).
static inline void pxa_pxq_dispatch_note(const char * where, int fu, int fg) {
    static const bool on = [](){ const char * e = getenv("PXA_PXQ_DISPATCH_DBG"); return e && atoi(e) != 0; }();
    if (!on) return;
    static std::atomic<uint64_t> seen{0};
    const int bit = (fu*9 + fg) & 63;
    const uint64_t m = 1ull << bit;
    if (seen.fetch_or(m) & m) return;
    fprintf(stderr, "PXA_PXQ_DISPATCH: %s fmt_up=%d fmt_gate=%d (3=P6 4=P6HQ 5=P2 6=P3 7=P6R 8=P1)\n", where, fu, fg);
}

// returns 0 if it handled the node, -1 to decline (caller falls through to the stock paths)
static int pxa_pxq_mmv_2d(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                          const ggml_tensor * src1, ggml_tensor * dst) {
    if (!pxa_pxq4_2d_enabled()) return -1;

    // PXA_PXQ_MMVQ: hand the dense decode GEMV to the q8_1 MMVQ kernel instead. Declining here
    // is what routes it -- ggml_cuda_mul_mat tries this driver first. Only the MMVQ batch window
    // (ny <= 8) is handed over; prefill keeps the bespoke 2D drivers untouched.
    if (pxa_pxq_mmvq_type(src0->type)
        && pxa_pxq_mmvq_on(ggml_cuda_info().devices[ctx.device].cc)
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE
        && src1->ne[2] == 1 && src1->ne[3] == 1
        && src0->ne[2] == 1 && src0->ne[3] == 1
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        return -1;
    }

    const int fmt = pxa_pxq_fmt(src0->type);
    if (fmt == PXA_PXQ_FMT_NONE) return -1;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return -1;

    // true 2D weight, single batch on both sides
    if (src0->ne[2] != 1 || src0->ne[3] != 1) return -1;
    if (src1->ne[2] != 1 || src1->ne[3] != 1) return -1;
    if (dst->ne[2]  != 1 || dst->ne[3]  != 1) return -1;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return -1;

    const int64_t K = src0->ne[0], R = src0->ne[1];
    if (K % PXQ6_QK || R % PXQ6_BM) return -1;      // slab width / 64-row panel height
    if (dst->ne[0] != R) return -1;

    const int64_t ny = src1->ne[1];                 // tokens this call
    if (ny < 1 || ny > pxa_pxq4_2d_max_ny()) return -1;
    if (src1->ne[0] != K) return -1;
    if (dst->ne[1] != ny) return -1;

    if (!pxa_pxq4_bufs_on_device(ctx, {src0, src1, dst})) return -1;

    // K1-2D x-STAGE CAP. The plain (S == 1) driver stages the WHOLE x vector in shared memory,
    // which caps K at (46K - KSEG*64*4)/4 = 11264 floats. A dense ffn_down is [17408, 5120], so
    // every layer of a dense model blew this cap and the node fell through to
    // k_pxq6_dequant_matrix + cuBLAS -- a full dequantize to f32 on EVERY token. Measured on
    // Qwable-27B (2xV100 decode): 3.35 t/s vs MXFP4's 36.3, and disabling this driver outright
    // costs a further 18x, which places the cost on that fallback. Invisible on MoE, where
    // ffn_down_exps is a 3D _exps tensor owned by the MoE grouped driver.
    //
    // The K8-2D S-split below already stages x in K/S-sized chunks and so fits any K we can
    // reach; it was simply unreachable, because this gate declined the node before it. Compute
    // the smallest S that fits rather than declining, and let the occupancy heuristic raise it
    // from there. S_min == 1 for every shape that passed before, so those shapes keep their
    // exact previous launch geometry and byte-reproduce.
    const int    kslabs   = (int)(K/PXQ6_QK);
    // PXQ_CANON_v1: NFIX fixed canonical chunks for this shape; any split S must divide it.
    const int    nfix     = pxq6_canon_nfix(kslabs, PXQ6_MMV_SPLIT_MAX);
    const size_t smem_fix = PXQ4_MMV_KSEG*64*sizeof(float);   // red[] — plain form only
    const size_t smem_cap = 46*1024;
    auto smem_for_S = [&](int S) {
        // split kernels stage only their x slice (the canon rewrite dropped their red[])
        return (size_t)((kslabs + S - 1)/S)*PXQ6_QK*sizeof(float) + (S == 1 ? smem_fix : 0);
    };
    int S_min = 1;
    while (S_min < PXQ6_MMV_SPLIT_MAX && smem_for_S(S_min) > smem_cap) S_min *= 2;
    if (smem_for_S(S_min) > smem_cap)     return -1;   // wider than even S=MAX can stage
    if (S_min > nfix)                     return -1;   // canonical chunking cannot cover this S
    const bool must_split = S_min > 1;                 // the plain form cannot stage this K
    if (must_split && !pxa_pxq4_2d_split()) return -1;

    const size_t smem = (size_t)K*sizeof(float) + smem_fix;   // plain form only (!must_split)

    // Belt (2026-07-25): the fused MoE drivers upload the env-override book/SUB16 tables and the
    // P2/P3 books on their first call, and every shipped file so far has a PXQ MoE node ahead of
    // its PXQ 2D nodes -- so this driver got them for free. An attention-only (or embd-only)
    // PXQ-native file has no such node: upload here too. No-op unless PXA_PXQ6_BOOK/_SUB/_SUB_HQ/
    // PXA_PXQ6R_BOOK is set, and idempotent per device.
    if (fmt >= PXA_PXQ_FMT_P6) pxq6_maybe_upload_tables(ctx.device);
    if (fmt >= PXA_PXQ_FMT_P2) pxq23_maybe_upload_books(ctx.device);

    const int  pair = pxa_pxq6_decode_mode();
    const bool vecx = pxa_pxq6_vecx();
    auto * kern = pxq6_pick_mmv(fmt, pair, vecx);
    if (!kern) return -1;                           // no instantiation for this format
    int nthr_mmv = 256;
    if (pxa_pxq6_rowx2()) {
        auto * k2 = pxq6_pick_mmv_x2(fmt, pair, vecx);
        pxa_pxq6_rowx2_log(0, k2 != nullptr);
        if (k2) { kern = k2; nthr_mmv = 128; }
    }
    if (nthr_mmv == 256 && (pxa_pxq6_qpf() & 1)) {   // Q1 yields to an engaged R2 twin
        auto * kq = pxq6_pick_mmv_qp(fmt, pair, vecx);
        pxa_pxq6_qpf_log(0, kq != nullptr);
        if (kq) kern = kq;
    }

    cudaStream_t stream = ctx.stream();

    // one-entry ids buffer == {0}; both strides zero so every (token, slot) block reads it.
    // Persistent per device; the pool form only mid-graph-capture.
    const int32_t * idz = pxa_pxq_zero_ids(ctx.device, stream);
    ggml_cuda_pool_alloc<int32_t> ids_pool(ctx.pool());
    if (!idz) {
        ids_pool.alloc(1);
        CUDA_CHECK(cudaMemsetAsync(ids_pool.get(), 0, sizeof(int32_t), stream));
        idz = ids_pool.get();
    }

    const int panels = (int)(R/PXQ4_BM);

    // K8-2D: S-way K-chunk split when the plain grid under-fills the machine (or its full-K
    // x stage caps occupancy). S = smallest power of two reaching the block target, capped by
    // NFIX (so S always divides the canonical chunking => bit-exact vs unsplit).
    if (pxa_pxq4_2d_split()) {
        const int target = pxa_pxq4_2d_split_target(ctx.device);

        // PXA_PXQ_MMV_TOK (2026-08-30): TOK-batched dense decode. On an ny>1 verify batch or
        // concurrent decode, fold TOK consecutive tokens into one block (grid.z = ceil(ny/TOK))
        // so the panel weight walk is shared across the group's tokens (kernel header in
        // pxq6.cuh). ws sizing and the canonical reducer are IDENTICAL to the per-token S-split
        // arm -- output is bit-identical; only the launch shape and the weight-read temporal
        // locality change. S is re-derived for this arm: the smem cap sees TOK slices, and the
        // occupancy target sees ny/TOK grid-z slices, so S both starts higher (fit) and climbs
        // higher (fill) than the per-token S. Declines (lever off, ny==1, no instantiation,
        // smem cap unmeetable) fall through to the per-token launches unchanged.
        const int tok_req = pxa_pxq_mmv_tok();
        if (tok_req >= 2 && ny >= 2) {
            const int tok = ny < (int64_t)tok_req ? (int)ny : tok_req;
            auto * kt = pxq6_pick_mmv_ksplit_gen_tok(fmt, pair, vecx, tok);
            auto smem_tok = [&](int Sx) {
                return (size_t)tok*(size_t)((kslabs + Sx - 1)/Sx)*PXQ6_QK*sizeof(float);
            };
            int St = S_min;
            while (St < nfix && smem_tok(St) > smem_cap) St *= 2;
            const int nyz = ((int)ny + tok - 1)/tok;
            while (St < nfix && (int64_t)panels*nyz*St < target) St *= 2;
            if (kt && tok >= 2 && St <= nfix && smem_tok(St) <= smem_cap) {
                const size_t need = (size_t)ny*(size_t)(PXQ6_MMV_SPLIT_MAX*PXQ4_MMV_KSEG)*(size_t)R;
                float * ws = pxq6_ksplit_workspace(ctx.device, stream, need);
                if (ws) {
                    const size_t smem_t = smem_tok(St);
                    dim3 gridt((unsigned)(panels*St), 1u, (unsigned)nyz);
                    pxa_pxq_mmv_tok_log(ctx.device, true, tok, gridt.x, gridt.y, gridt.z,
                                        (int)ny, St, smem_t, fmt, pair, (int)vecx);
                    kt<<<gridt, 256, smem_t, stream>>>(
                        (const uint8_t *)src0->data,
                        (const char *)src1->data, src1->nb[1], /* x_slot_stride */ 0,
                        ws,
                        (const char *)idz, /* ids_nb0 */ 0, /* ids_nb1 */ 0,
                        (int)R, (int)K, /* n_as */ 1, /* n_ids */ 1, St, (int)ny);
                    CUDA_CHECK(cudaGetLastError());
                    const int rblk = pxa_pxq_reduce_blk();
                    dim3 gridr((unsigned)((R + rblk - 1)/rblk), 1u, (unsigned)ny);
                    k_pxq_mmv_reduce_s<<<gridr, rblk, 0, stream>>>(ws,
                        (char *)dst->data, dst->nb[1], /* dst_slot_stride */ 0,
                        (const char *)idz, /* ids_nb0 */ 0, /* ids_nb1 */ 0,
                        (int)R, /* n_as */ 1, /* n_ids */ 1, kslabs);
                    CUDA_CHECK(cudaGetLastError());
                    return 0;
                }
            } else if (tok >= 2) {
                pxa_pxq_mmv_tok_log(ctx.device, false, tok, 0, 0, 0,
                                    (int)ny, St, smem_tok(St), fmt, pair, (int)vecx);
            }
        }

        int S = S_min;               // >= 1; above 1 only when the x stage demands it
        while (S < nfix && (int64_t)panels*ny*S < target) S *= 2;
        if (S > 1) {
            auto * ksg = pxq6_pick_mmv_ksplit_gen(fmt, pair, vecx);
            int nthr_ksg = 256;
            if (ksg && pxa_pxq6_rowx2()) {
                auto * k2 = pxq6_pick_mmv_ksplit_gen_x2(fmt, pair, vecx);
                pxa_pxq6_rowx2_log(1, k2 != nullptr);
                if (k2) { ksg = k2; nthr_ksg = 128; }
            }
            if (ksg && nthr_ksg == 256 && (pxa_pxq6_qpf() & 1)) {   // Q1 yields to an engaged R2 twin
                auto * kq = pxq6_pick_mmv_ksplit_gen_qp(fmt, pair, vecx);
                pxa_pxq6_qpf_log(1, kq != nullptr);
                if (kq) ksg = kq;
            }
            if (ksg) {
                // PXQ_CANON_v1 workspace: raw per-(fixed-chunk, lane) partials
                const size_t need = (size_t)ny*(size_t)(PXQ6_MMV_SPLIT_MAX*PXQ4_MMV_KSEG)*(size_t)R;
                float * ws = pxq6_ksplit_workspace(ctx.device, stream, need);
                pxa_pxq4_2d_split_log(ctx.device, ws != nullptr, S, panels, (int)ny, (int)R, (int)K);
                if (ws) {
                    int * ctrs = pxa_pxq_split_fusered()
                        ? pxq6_ksplit_counters(ctx.device, stream) : nullptr;
                    const int kc_max = (kslabs + S - 1)/S;
                    const size_t smem_s = (size_t)kc_max*PXQ6_QK*sizeof(float);
                    dim3 grids((unsigned)(panels*S), 1u, (unsigned)ny);
                    ksg<<<grids, nthr_ksg, smem_s, stream>>>(
                        (const uint8_t *)src0->data,
                        (const char *)src1->data, src1->nb[1], /* x_slot_stride */ 0,
                        ws,
                        (const char *)idz, /* ids_nb0 */ 0, /* ids_nb1 */ 0,
                        (char *)dst->data, dst->nb[1], /* dst_slot_stride */ 0, ctrs,
                        (int)R, (int)K, /* n_as */ 1, /* n_ids */ 1, S);
                    CUDA_CHECK(cudaGetLastError());
                    if (!ctrs) {
                        const int rblk = pxa_pxq_reduce_blk();
                        dim3 gridr((unsigned)((R + rblk - 1)/rblk), 1u, (unsigned)ny);
                        k_pxq_mmv_reduce_s<<<gridr, rblk, 0, stream>>>(ws,
                            (char *)dst->data, dst->nb[1], /* dst_slot_stride */ 0,
                            (const char *)idz, /* ids_nb0 */ 0, /* ids_nb1 */ 0,
                            (int)R, /* n_as */ 1, /* n_ids */ 1, kslabs);
                        CUDA_CHECK(cudaGetLastError());
                    }
                    return 0;
                }
            }
        }
    }

    // K1-2D: re-grid when the plain grid under-fills the machine. Bit-identical; every decline
    // (no instantiation / workspace unavailable under graph capture) falls through to the
    // single-launch form below, which produces the same bytes.
    // Both remaining forms stage the full K (smem_ks = K*4 below, smem in the plain launch),
    // so neither can serve a must_split shape. Decline instead: dequant+cuBLAS is 18x slower
    // than this driver but it is correct, whereas launching either of these at 68 KB is not.
    if (must_split) return -1;

    if (pxa_pxq4_2d_ksplit() && (int64_t)panels*ny < (int64_t)pxa_pxq4_2d_ksplit_minblk(ctx.device)) {
        auto * ks = pxq6_pick_mmv_ksplit(fmt, pair, vecx);
        if (ks) {
            const size_t need = (size_t)ny*PXQ4_MMV_KSEG*(size_t)R;
            float * ws = pxq6_ksplit_workspace(ctx.device, stream, need);
            pxa_pxq4_2d_ksplit_log(ctx.device, ws != nullptr, panels, (int)ny, (int)R, (int)K);
            if (ws) {
                const size_t smem_ks = (size_t)K*sizeof(float);   // no red[] in the 64-thr form
                dim3 grids((unsigned)(panels*PXQ4_MMV_KSEG), 1u, (unsigned)ny);
                ks<<<grids, 64, smem_ks, stream>>>(
                    (const uint8_t *)src0->data,
                    (const char *)src1->data, src1->nb[1], /* x_slot_stride */ 0,
                    ws,
                    (const char *)idz, /* ids_nb0 */ 0, /* ids_nb1 */ 0,
                    (int)R, (int)K, /* n_as */ 1, /* n_ids */ 1);
                CUDA_CHECK(cudaGetLastError());
                const int rblk = pxa_pxq_reduce_blk();
                        dim3 gridr((unsigned)((R + rblk - 1)/rblk), 1u, (unsigned)ny);
                k_pxq_mmv_reduce<<<gridr, rblk, 0, stream>>>(ws,
                    (char *)dst->data, dst->nb[1], /* dst_slot_stride */ 0,
                    (const char *)idz, /* ids_nb0 */ 0, /* ids_nb1 */ 0,
                    (int)R, /* n_as */ 1, /* n_ids */ 1);
                CUDA_CHECK(cudaGetLastError());
                return 0;
            }
        }
    }

    dim3 grid((unsigned)panels, 1u, (unsigned)ny);
    kern<<<grid, nthr_mmv, smem, stream>>>(
        (const uint8_t *)src0->data,
        (const char *)src1->data, src1->nb[1], /* x_slot_stride  */ 0,
        (char *)dst->data,        dst->nb[1],  /* dst_slot_stride*/ 0,
        (const char *)idz, /* ids_nb0 */ 0, /* ids_nb1 */ 0,
        (int)R, (int)K, /* n_as */ 1);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================ PXQ 2D prefill GEMM (non-MoE MUL_MAT) ============================
// Companion to pxa_pxq_mmv_2d for the ny > PXA_PXQ4_2D_MAX_NY window. A plain 2D PXQ weight is
// the ONE-EXPERT case of the expert-stacked grouped GEMM: reuse k_pxq6_gemm_grouped verbatim with
// a tile map whose every entry carries e == 0, so pxq6_panel() degenerates to W + p*panel_stride
// -- the same degeneration the mmv driver gets from its one-entry zero ids buffer. C needs no
// scatter (Ct = C + row0*R + p*64 with Ct[t*R + row] IS the contiguous f32 dst laid out [ny][R])
// and a plain 2D MUL_MAT never carries the MoE fused-bias sources, so bias is always nullptr.
//
// ARCH GATE, host-side compute capability (dispatch is a host decision; __CUDA_ARCH__ is
// unavailable and meaningless here on a mixed-card rig where ONE binary drives sm_60 and sm_70):
//   PXA_PXQ_GEMM_2D=0  OFF everywhere (default -- see the measurement below)
//   PXA_PXQ_GEMM_2D=1  sm_60 only (GP100: full-rate fp16, no tensor cores)
//   PXA_PXQ_GEMM_2D=2  sm_60 + sm_70 (any fast-fp16 pre-Turing arch)
//
// MEASURED 2026-07-25, Fusion4-35B-PXQ4N-attn, fill 5739 / n_predict 256 / temp 0, n=8 per arm,
// interleaved in 4 blocks with a fresh server per arm per block, `-c 8192 -b 2048 -ub 2048`:
//   2xV100 -ts 1.05,0.95, mode 2 : prefill 1718.3 -> 1757.9 t/s = +2.30%, decode flat (+0.4%)
//   4xP100 -ts 1,1,1,1,  mode 1 : prefill  799.6 ->  806.7 t/s = +0.89%, decode flat (-0.03%)
// Complete rank separation within both the cold-rep and warm-rep strata on both arches, so the
// direction is not noise -- but both are BELOW the +3% pre-registered keep line, hence default 0.
//
// >>> SUPERSEDED ON sm_70 (2026-07-28). The +2.30% above was measured against the PRE-COALESCING
// k_pxq6_dequant_matrix, which stored one thread per row -- 32 sectors for 64 useful bytes -- so
// the incumbent this beat no longer exists. Re-measured against the coalesced dequant on the same
// 2xV100 cell: mode 2 is -18.6% on DENSE, and on MoE it is inside run-to-run noise (~1.8% here).
// A lever's value is relative to the incumbent it replaces, and that incumbent got fixed.
// Mode 2 is therefore CLAMPED to sm_60 below, same as mode 1, and says so once on stderr rather
// than silently regressing. sm_60 is untouched: the P100 number (+35% dense prefill on the
// coalesced binary) is why this code still ships at all. <<<
//
// The pre-implementation spec predicted a MULTI-X REGRESSION on sm_70 and had this gated to sm_60
// only. That prediction is wrong, and the reason is worth keeping: it compared this half2 tile
// (~15-20 TF) against cuBLAS HMMA (~90 TF) alone, but the incumbent is not cuBLAS alone -- it is
// k_pxq6_dequant_matrix (scalar 2-byte stores at a row stride of K, ~1/16 store efficiency)
// PLUS cuBLAS. Folding the dequant into the GEMM pays for the slower matmul with ~2% left over.
// The corollary for whoever tackles the prefill deficit next: this claims BOTH halves of the
// incumbent and nets only +2.3%, so a fix that coalesces the dequant and KEEPS cuBLAS's HMMA
// GEMM should dominate this on sm_70. Decode is untouched either way (ny <= PXA_PXQ4_2D_MAX_NY
// is declined here and owned by pxa_pxq_mmv_2d).
// gemm_2d ENHANCE auto (2026-07-29, model-aware): mode 1 (sm_60 only) when an sm_60 device is
// present AND the loaded model is DENSE AND it carries PXQ tensors. Basis: +35% P100 dense
// prefill on the coalesced binary (2026-07-28 arch-split note above). The sm_60 x MoE cell was
// only ever measured against the PRE-coalescing incumbent (+0.89% prefill, decode flat) — post-
// coalescing it is UNMEASURED and the sm_70 dense flip (-18.6%) shows the coalescing fix can
// invert a win, so MoE stays OFF until someone measures it. Explicit env always wins; cached
// against the model-profile generation (profile can land after CUDA init in the server flow).
static int pxa_pxq_gemm_2d_mode() {
    static int cached_gen = -1;
    static int cached     = 0;
    const int gen = ggml_pxa_model_profile_generation();
    if (cached_gen == gen) return cached;
    const char * e = getenv("PXA_PXQ_GEMM_2D");
    int m = e ? atoi(e) : 0;
    if (!e && pxa_config_level() == 2 && g_pxa_topology.valid && g_pxa_topology.has_sm60
        && pxa_model_known() && !pxa_model_is_moe() && pxa_model().n_pxq_mmvq_tensors > 0) {
        m = 1;
        fprintf(stderr, "PXA_AUTO: PXQ_GEMM_2D=1 (ENHANCE x sm_60 present x dense PXQ model: "
                        "+35%% P100 dense prefill measured 2026-07-28; MoE stays off — post-"
                        "coalescing sm_60 MoE cell unmeasured; override PXA_PXQ_GEMM_2D)\n");
    }
    if (m < 0 || m > 2) m = 0;
    cached = m;
    cached_gen = gen;
    return m;
}

// The claim is silent when it declines, and a decline is indistinguishable from a firing that did
// nothing. Log the first firing and the first decline per device, like the KSPLIT lever does.
static void pxa_pxq_gemm_2d_log(int device, bool fired, int R, int K, int ny, int cc) {
    static std::atomic<uint32_t> seen{0};
    if (device < 0 || device >= 16) return;
    const uint32_t bit = 1u << (2*device + (fired ? 0 : 1));
    if (seen.fetch_or(bit) & bit) return;
    fprintf(stderr, "PXA_PXQ_GEMM_2D dev%d: %s (mode=%d cc=%d R=%d K=%d ny=%d)\n", device,
            fired ? "FIRING" : "DECLINED -> dequant+cuBLAS", pxa_pxq_gemm_2d_mode(), cc, R, K, ny);
}

// returns 0 if it handled the node, -1 to decline (caller falls through to the stock paths)
// =================================================================================================
// PXA_X_CACHE (2026-09-01) — hoist the shared activation conversion across the GEMMs of one block.
//
// Translated from shinbunbun `mmvq-q8-1-activation-cache`, which caches the q8_1 quantization of
// the activation row across the many stock-MMVQ calls that share it in one layer. AUDIT of our
// decode path first, because most of what that patch fixes does not exist here:
//
//   * The PXQ decode mmv family (k_pxq6_mmv*, k_pxq6_mmv_ksplit_gen, the fused gateup) reads f32
//     activations DIRECTLY and stages them in shared memory per block. There is no quantization
//     or dtype conversion of x anywhere on that path, so there is nothing to hoist.
//   * pxa_pxq4_moe_fast_tg converts nothing and launches ONE grid over (R/BM) x n_ids x Ny, so the
//     routed expert GEMVs of a layer already share a single pass over x by construction.
//   * The stock q8_1 fast-TG MoE branch already quantizes dst once for all n_ids*Ny rows
//     (ggml-cuda.cu ~:5894) — upstream had already done this hoist for the path shinbunbun patched.
//   * The hc_* f16 GEMVs (ggml_cuda_small_gemv_f16 / ggml_cuda_wide_gemv_f16) consume f32 x with
//     no conversion at all; only the cuBLAS chain they REPLACE converted per call.
//
// The one place the mechanism does map is PREFILL: pxa_pxq_gemm_2d converts src1 f32 -> f16 into a
// fresh pool buffer on EVERY call, and the q, k and v projections of an attention block are three
// consecutive MUL_MAT nodes over the identical `cur` tensor. At ub 2048 with K = n_embd that is
// 2048*2560*4 B read + 10 MB written, three times, for one distinct result — two thirds redundant.
//
// SAFETY. The cache is keyed on the src1 tensor OBJECT, its data pointer, and ny/K, and — the
// load-bearing part — it is only honoured when the previous conversion happened on the
// IMMEDIATELY PRECEDING executed graph node (the op_seq check below). That rules out any
// intervening node having written through an alias of src1's buffer, which a pointer-only key
// could not: ggml pool addresses are recycled within a graph and in-place ops mutate through views.
//
// Bit-identical when it fires: the reused buffer is the output of the same converter over the same
// bytes, so the kernel sees exactly the activations it would have converted itself. Default OFF.
// =================================================================================================
static std::atomic<uint64_t> pxa_op_seq{0};   // bumped once per executed graph node

static inline bool pxa_x_cache_on() {
    static const bool v = [] {
        const char * e = getenv("PXA_X_CACHE");
        const bool on = e && atoi(e) != 0;
        if (on) fprintf(stderr, "PXA_X_CACHE: armed — the f32->f16 activation stage of the PXQ 2D "
                                "prefill GEMM is reused across consecutive nodes that share src1 "
                                "(bit-identical); PXA_X_CACHE=0 reverts\n");
        return on;
    }();
    return v;
}

struct pxa_x_cache_t {
    half * ptr = nullptr;
    size_t sz  = 0;                       // bytes
    const ggml_tensor * key_t = nullptr;
    const void * key_d = nullptr;
    int64_t key_ny = 0, key_K = 0;
    uint64_t key_seq = 0;                 // pxa_op_seq at the time of the conversion
};

static inline half * pxa_x_cache_get(int device, cudaStream_t stream,
                                     const ggml_tensor * src1, int64_t ny, int64_t K, bool & hit) {
    static pxa_x_cache_t cc_[64];
    hit = false;
    if (device < 0 || device >= 64) return nullptr;
    pxa_x_cache_t & c = cc_[device];
    const uint64_t seq = pxa_op_seq.load(std::memory_order_relaxed);

    if (c.ptr && c.key_t == src1 && c.key_d == src1->data &&
        c.key_ny == ny && c.key_K == K && c.key_seq + 1 == seq) {
        c.key_seq = seq;                  // chain q -> k -> v
        hit = true;
        return c.ptr;
    }

    const size_t need = (size_t)ny*K*sizeof(half);
    if (c.sz < need) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &st);
        if (st != cudaStreamCaptureStatusNone) return nullptr;   // cannot grow mid-capture
        if (c.ptr) cudaFree(c.ptr);
        c.ptr = nullptr; c.sz = 0;
        if (cudaMalloc(&c.ptr, need) != cudaSuccess) { c.ptr = nullptr; (void)cudaGetLastError(); return nullptr; }
        c.sz = need;
    }
    c.key_t = src1; c.key_d = src1->data; c.key_ny = ny; c.key_K = K; c.key_seq = seq;
    return c.ptr;
}

static inline void pxa_x_cache_log(int device, bool hit, int64_t ny, int64_t K) {
    static std::atomic<long> hits{0}, misses{0};
    const long h = hit ? hits.fetch_add(1) + 1 : hits.load();
    const long m = hit ? misses.load() : misses.fetch_add(1) + 1;
    if (((h + m) % 512) == 0 || (h + m) < 4) {
        fprintf(stderr, "PXA_X_CACHE: dev=%d ny=%d K=%d reuse=%ld convert=%ld\n",
                device, (int) ny, (int) K, h, m);
    }
}

// =================================================================================================
// PXA_PXQ4_MMQ_ACACHE — the int8 twin of PXA_X_CACHE, for the dp4a MMQ prefill tile.
//
// Stock MMQ quantizes the ubatch's activations ONCE (quantize_mmq_q8_1 on src1) and every weight
// in the graph then reads that one buffer; our driver is per-MUL_MAT, so without this the q, k and
// v projections would each re-quantize the identical `cur`, and gate/up would do it twice more.
// That is pure overhead against the dequant saving the tile exists to collect (main #282 note 2).
//
// Same key and same safety argument as PXA_X_CACHE: the src1 tensor OBJECT, its data pointer,
// ny/K, AND the requirement that the previous stage ran on the IMMEDIATELY preceding executed
// graph node, which is what rules out an intervening write through an alias of src1's buffer.
// Bit-identical when it fires (same quantizer, same bytes). PXA_PXQ4_MMQ_ACACHE=0 reverts to
// staging on every call. Buffers are cudaMalloc'd (not pool) so they outlive the node.
// =================================================================================================
struct pxa_aq_cache_t {
    uint8_t * q   = nullptr;
    float   * d   = nullptr;
    size_t    qsz = 0, dsz = 0;
    const ggml_tensor * key_t = nullptr;
    const void * key_d = nullptr;
    int64_t key_ny = 0, key_K = 0;
    uint64_t key_seq = 0;
};

static inline bool pxa_aq_cache_on() {
    static const bool v = [] {
        const char * e = getenv("PXA_PXQ4_MMQ_ACACHE");
        return !(e && atoi(e) == 0);
    }();
    return v;
}

// returns true on a HIT (q/d already hold this src1's stage); false means the caller must stage.
// q/d are set in both cases unless the cache declines entirely (then both stay null).
static inline bool pxa_aq_cache_get(int device, cudaStream_t stream, const ggml_tensor * src1,
                                    int64_t ny, int64_t K, uint8_t *& q, float *& d) {
    static pxa_aq_cache_t cc_[64];
    q = nullptr; d = nullptr;
    if (device < 0 || device >= 64) return false;
    pxa_aq_cache_t & c = cc_[device];
    const uint64_t seq = pxa_op_seq.load(std::memory_order_relaxed);

    if (c.q && c.key_t == src1 && c.key_d == src1->data &&
        c.key_ny == ny && c.key_K == K && c.key_seq + 1 == seq) {
        c.key_seq = seq;                  // chain q -> k -> v / gate -> up
        q = c.q; d = c.d;
        return true;
    }

    const size_t needq = (size_t)ny*K;
    const size_t needd = (size_t)ny*(K/PXQ4_QK)*sizeof(float);
    if (c.qsz < needq || c.dsz < needd) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &st);
        if (st != cudaStreamCaptureStatusNone) return false;     // cannot grow mid-capture
        if (c.q) cudaFree(c.q);
        if (c.d) cudaFree(c.d);
        c.q = nullptr; c.d = nullptr; c.qsz = 0; c.dsz = 0;
        if (cudaMalloc(&c.q, needq) != cudaSuccess) { c.q = nullptr; (void)cudaGetLastError(); return false; }
        if (cudaMalloc(&c.d, needd) != cudaSuccess) { cudaFree(c.q); c.q = nullptr; c.d = nullptr; (void)cudaGetLastError(); return false; }
        c.qsz = needq; c.dsz = needd;
    }
    c.key_t = src1; c.key_d = src1->data; c.key_ny = ny; c.key_K = K; c.key_seq = seq;
    q = c.q; d = c.d;
    return false;
}

static inline void pxa_aq_cache_log(int device, bool hit, int64_t ny, int64_t K) {
    static std::atomic<long> hits{0}, misses{0};
    const long h = hit ? hits.fetch_add(1) + 1 : hits.load();
    const long m = hit ? misses.load() : misses.fetch_add(1) + 1;
    if (((h + m) % 2048) == 0 || (h + m) < 4) {
        fprintf(stderr, "PXA_PXQ4_MMQ_ACACHE: dev=%d ny=%d K=%d reuse=%ld stage=%ld\n",
                device, (int) ny, (int) K, h, m);
    }
}

static int pxa_pxq_gemm_2d(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                           const ggml_tensor * src1, ggml_tensor * dst) {
    const int mode = pxa_pxq_gemm_2d_mode();
    // V70 (sm_70 register-direct m8n8k4 PXQ4 GEMM, pxq4-v70.cuh) has its OWN env gate and is not
    // a mode of the sm_60 half2 tile, so it must not be shut out by PXA_PXQ_GEMM_2D defaulting
    // to 0 on a Volta-only node. Everything below the shared shape checks re-splits on v70_env.
    const bool v70_env = pxa_pxq_gemm_v70();
    // M1 (pxq4-mmq.cuh) is likewise a route of its own, not a mode of the sm_60 half2 tile:
    // it must stay reachable when PXA_PXQ_GEMM_2D is unset (its normal state on a Volta node).
    const int  mmq_env = pxa_pxq4_mmq();
    if (mode == 0 && !v70_env && !mmq_env) return -1;

    const int fmt = pxa_pxq_fmt(src0->type);
    if (fmt == PXA_PXQ_FMT_NONE) return -1;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return -1;
    // a 2D MUL_MAT CAN carry GGML_PREC_F32 (the cuBLAS fallback honours it by taking the fp32
    // SGEMM branch). This tile accumulates in half2 -- never silently downgrade such a node.
    // (MUL_MAT_ID nodes never set it, which is why the MoE driver has no equivalent check.)
    if (dst->op_params[0] != GGML_PREC_DEFAULT) return -1;

    if (src0->ne[2] != 1 || src0->ne[3] != 1) return -1;      // true 2D weight
    if (src1->ne[2] != 1 || src1->ne[3] != 1) return -1;
    if (dst->ne[2]  != 1 || dst->ne[3]  != 1) return -1;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return -1;

    const int64_t K = src0->ne[0], R = src0->ne[1];
    if (K % PXQ6_QK || R % PXQ6_BM) return -1;                // slab width / 64-row panel height
    if (dst->ne[0] != R || src1->ne[0] != K) return -1;
    const int64_t ny = src1->ne[1];
    if (dst->ne[1] != ny) return -1;
    if (ny <= pxa_pxq4_2d_max_ny()) return -1;                // the decode mmv driver owns small ny

    if (!pxa_pxq4_bufs_on_device(ctx, {src0, src1, dst})) return -1;
    if (src0->buffer && ggml_backend_buffer_is_cuda_split(src0->buffer)) return -1;

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    // Both modes are sm_60-only as of 2026-07-28: mode 2's sm_70 admission was justified by a
    // measurement against the pre-coalescing dequant, and re-measuring flipped it to -18.6%.
    // Mode 2 is kept as an accepted value (rather than rejected) so existing scripts still run,
    // but it is clamped here and announced once per device so the clamp is never silent.
    if (mode == 2 && cc >= CC_VOLTA) {
        static std::atomic<bool> told{false};
        if (!told.exchange(true)) {
            fprintf(stderr, "PXA_PXQ_GEMM_2D=2 clamped to sm_60 on cc=%d "
                    "(measured -18.6%% on sm_70 against the coalesced dequant)\n", cc);
        }
    }
    // V70: sm_70 exactly, PXQ4 core tier, above its own ny floor, and not denied by the route
    // table. It replaces the ENTIRE incumbent route (k_pxq6_dequant_matrix -> fp16 pool scratch
    // -> cublasGemmEx), so a shape that does not beat that route including its dequant is marked
    // ROUTE_CUBLAS in pxq4_v70_route_deny and falls through here to the generic path unchanged.
    const int  v70_cfg = pxa_pxq_v70_cfg();
    const bool use_v70 = v70_env
                      && cc == GGML_CUDA_CC_VOLTA
                      && pxa_ml_volta_mma_available(cc)
                      && src0->type == GGML_TYPE_PXQ4
                      && ny >= pxa_pxq_v70_min_ny()
                      && (R % pxa_pxq_v70_bm(v70_cfg)) == 0
                      && pxa_pxq_v70_route_ok((int)R, (int)K, (int)ny);

    // mode != 0 is part of arch_ok, NOT just the early return: the early return above was
    // relaxed so V70 (which has its own gate) stays reachable when PXA_PXQ_GEMM_2D is unset,
    // and without this term an armed V70 would ALSO switch the sm_60/61 generic tile on for
    // every device on a mixed node. With the V70 gate off this term restores the old path
    // exactly; with it on, V70 is the only thing mode == 0 lets through.
    // M1: dp4a MMQ tile. 4-bit nibble tiers only (PXQ4 / PXQ4HQ), prefill-sized token counts,
    // and a card with real IDP4A (sm_61+ / sm_70; cc 600 emulates it and is declined unless
    // PXA_PXQ4_MMQ=2 forces the test path). It replaces the ENTIRE incumbent route
    // (k_pxq6_dequant_matrix -> fp16 pool scratch -> cublasGemmEx) for the node it claims.
    const bool mmq_arch = (cc >= 610) || (mmq_env == 2);
    const bool use_mmq  = mmq_env != 0
                       && !use_v70
                       && mmq_arch
                       && (fmt == PXA_PXQ_FMT_P6 || fmt == PXA_PXQ_FMT_P6HQ)
                       && ny >= pxa_pxq4_mmq_min_ny()
                       && pxa_pxq4_mmq_route_ok((int)R, (int)K);

    const bool arch_ok = mode != 0 && cc < CC_VOLTA && fast_fp16_available(cc);
    if (!arch_ok && !use_v70 && !use_mmq) { pxa_pxq_gemm_2d_log(ctx.device, false, (int)R, (int)K, (int)ny, cc); return -1; }

    if (use_mmq) {
        const pxq4mmq_kern mk = pxq4mmq_pick(fmt);
        if (!mk.fn) { pxa_pxq_gemm_2d_log(ctx.device, false, (int)R, (int)K, (int)ny, cc); return -1; }
        const int64_t mtiles = (ny + PXQ4_BN - 1)/PXQ4_BN;
        if (mtiles > 65535) return -1;
        if (fmt >= PXA_PXQ_FMT_P6) pxq6_maybe_upload_tables(ctx.device);

        cudaStream_t st = ctx.stream();
        ggml_cuda_pool_alloc<pxq4_tile_info> mtile_info(ctx.pool(), (size_t)mtiles);
        k_pxq_tiles_2d<<<(unsigned)((mtiles + 255)/256), 256, 0, st>>>(
            mtile_info.get(), (int)ny, (int)mtiles);
        CUDA_CHECK(cudaGetLastError());

        // activation stage, reused across consecutive nodes that share src1 (q/k/v, gate/up)
        ggml_cuda_pool_alloc<uint8_t> Aq_pool;
        ggml_cuda_pool_alloc<float>   Ad_pool;
        uint8_t * Aq = nullptr; float * Ad = nullptr;
        bool ahit = false;
        if (pxa_aq_cache_on()) {
            ahit = pxa_aq_cache_get(ctx.device, st, src1, ny, K, Aq, Ad);
            if (Aq) pxa_aq_cache_log(ctx.device, ahit, ny, K);
        }
        if (!Aq) {
            Aq_pool.alloc(ctx.pool(), (size_t)ny*K);
            Ad_pool.alloc(ctx.pool(), (size_t)ny*(K/PXQ4_QK));
            Aq = Aq_pool.get(); Ad = Ad_pool.get();
            ahit = false;
        }
        if (!ahit) {
            k_pxq4mmq_quant_rows<<<(unsigned)ny, 128, 0, st>>>(
                (const float *)src1->data, Aq, Ad, (int)K);
            CUDA_CHECK(cudaGetLastError());
        }

        pxa_pxq_gemm_2d_log(ctx.device, true, (int)R, (int)K, (int)ny, cc);
        dim3 mgrid((unsigned)(R/PXQ4_BM), (unsigned)mtiles);
        mk.fn<<<mgrid, mk.block, 0, st>>>((const uint8_t *)src0->data, Aq, Ad,
                                          (float *)dst->data, mtile_info.get(), (int)R, (int)K);
        CUDA_CHECK(cudaGetLastError());
        return 0;
    }


    const int64_t ntiles = (ny + PXQ4_BN - 1)/PXQ4_BN;
    if (ntiles > 65535) return -1;                            // grid.y limit

    // env-override books: same belt as the mmv driver (see there).
    if (fmt >= PXA_PXQ_FMT_P6) pxq6_maybe_upload_tables(ctx.device);
    if (fmt >= PXA_PXQ_FMT_P2) pxq23_maybe_upload_books(ctx.device);

    pxq6_gemm_fn kern = use_v70 ? nullptr : pxq6_pick_gemm(fmt, pxa_pxq6_ragtail(), pxa_pxq6_pipe());
    if (!kern && !use_v70) return -1;

    cudaStream_t stream = ctx.stream();

    // V70 needs no tile-info array: its token tail is handled in-kernel (zero-filled B fragments
    // + a predicated epilogue), so skip dev_tiles / k_pxq_tiles_2d entirely.
    ggml_cuda_pool_alloc<pxq4_tile_info> dev_tiles;
    if (!use_v70) {
        dev_tiles.alloc(ctx.pool(), (size_t)ntiles);
        k_pxq_tiles_2d<<<(unsigned)((ntiles + 255)/256), 256, 0, stream>>>(
            dev_tiles.get(), (int)ny, (int)ntiles);
        CUDA_CHECK(cudaGetLastError());
    }

    // src1 contiguous f32 [ny][K] -> the row-major half [ny][K] the kernel indexes as At + srow*K.
    // Same converter, same argument order, as the cuBLAS fallback's src1 stage, so the kernel sees
    // byte-identical activations to the path it replaces.
    ggml_cuda_pool_alloc<half> A_f16;
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    GGML_ASSERT(to_fp16 != nullptr);

    // PXA_X_CACHE: reuse the stage when the previous graph node converted the identical src1.
    half * A = nullptr;
    if (pxa_x_cache_on()) {
        bool hit = false;
        half * cached = pxa_x_cache_get(ctx.device, stream, src1, ny, K, hit);
        if (cached) {
            if (!hit) to_fp16(src1->data, cached, ny, K, stream);
            pxa_x_cache_log(ctx.device, hit, ny, K);
            A = cached;
        }
    }
    if (!A) {
        A_f16.alloc(ctx.pool(), (size_t)ny*K);
        to_fp16(src1->data, A_f16.get(), ny, K, stream);
        A = A_f16.get();
    }

    pxa_pxq_gemm_2d_log(ctx.device, true, (int)R, (int)K, (int)ny, cc);

    if (use_v70) {
        if (pxa_pxq4_gemm_v70_launch((const uint8_t *)src0->data, A, (float *)dst->data,
                                     (int)R, (int)K, (int)ny, v70_cfg, stream) != 0) {
            return -1;   // shape not launchable for this config: hand the node back untouched
        }
        CUDA_CHECK(cudaGetLastError());
        return 0;
    }

    dim3 grid((unsigned)(R/PXQ4_BM), (unsigned)ntiles);
    kern<<<grid, 64, 0, stream>>>((const uint8_t *)src0->data, A, (float *)dst->data,
                                  /* bias */ nullptr, /* bias_nb1 */ 0,
                                  dev_tiles.get(), (int)R, (int)K);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// decode / fast-TG: ny <= 8 tokens, one fused mmv per (token, routed expert slot).
// All PXA slab formats (PXQ4/PXQ4-HQ/PXQ2/PXQ3/PXQ6) dispatch here on the policy kernel
// family. (The legacy id-250/id-251 old-family kernels were removed 2026-07-21.) Fastpath gates:
// K1 PXA_PXQ6_KSPLIT (bit-exact) / PXA_PXQ6_KSPLIT_GEN=S (G3-gated), K2 PXA_PXQ6_PAIRLUT /
// PXA_PXQ6_VECX (bit-exact) — all format-independent via the pxq6 policy family.
static int pxa_pxq4_moe_fast_tg(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cgraph * graph, int i) {
    ggml_tensor * next = i + 1 < graph->n_nodes ? graph->nodes[i+1] : nullptr;
    const ggml_tensor * src0_1 = dst->src[0];   // up   (bias = dst->src[4])
    const ggml_tensor * src0_2 = dst->src[1];   // gate (bias = dst->src[5])
    const ggml_tensor * src1   = dst->src[2];
    const ggml_tensor * ids    = dst->src[3];

    const int fmt   = pxa_pxq_fmt(src0_1->type);                       // up
    const int fmt_g = src0_2 ? pxa_pxq_fmt(src0_2->type) : PXA_PXQ_FMT_NONE;  // gate
    if (!src0_2 || fmt == PXA_PXQ_FMT_NONE || fmt_g == PXA_PXQ_FMT_NONE) return -1;
    pxa_pxq_dispatch_note("moe_gateup_decode", fmt, fmt_g);
    if (fmt >= PXA_PXQ_FMT_P6 || fmt_g >= PXA_PXQ_FMT_P6) pxq6_maybe_upload_tables(ctx.device);
    if (fmt >= PXA_PXQ_FMT_P2 || fmt_g >= PXA_PXQ_FMT_P2) pxq23_maybe_upload_books(ctx.device);
    const ggml_unary_op uop = (ggml_unary_op)dst->op_params[0];
    if (uop != GGML_UNARY_OP_SWIGLU_OAI && uop != GGML_UNARY_OP_SILU) return -1;
    if (src1->type != GGML_TYPE_F32) return -1;
    const int64_t R = src0_1->ne[1], K = src0_1->ne[0];
    if (R % PXQ4_BM || K % PXQ4_QK || src0_2->ne[0] != K || src0_2->ne[1] != R) return -1;
    if (!pxa_pxq4_bufs_on_device(ctx, {src0_1, src0_2, src1, dst})) return -1;
    const size_t smem_gu = (size_t)K*sizeof(float) + 2*PXQ4_MMV_KSEG*64*sizeof(float);
    if (smem_gu > 46*1024) return -1;

    cudaStream_t stream = ctx.stream();
    const int64_t n_as  = src0_1->ne[2];

    // PXA_E2PXQ (2026-08-04): the ORIGINAL PXA_E2 union probe lives inside the stock q8_1 fast-TG
    // branch, which excludes every PXQ type -- so it is unreachable on PXQ models and produced
    // ZERO output on the 122B. Mirrored here so the number can actually be taken.
    // Read-only. Decides TOK-batched verify: union(k) ~ k*n_ids means disjoint routing and an
    // irreducible per-token weight re-read; union(k) << k*n_ids means the panel can be shared.
    if (getenv("PXA_MOE_DEBUG")) {
        const ggml_tensor * pids = dst->src[3];
        const int64_t pNy   = src1->ne[2];
        const int64_t pnids = pids ? pids->ne[0] : 0;
        if (pNy > 1 && pNy <= 8 && pnids > 0 && pnids <= 32) {
            static int    e2_calls = 0;
            static double e2_union[9] = {0};
            static long   e2_cnt[9]   = {0};
            const int nids = (int)pnids;
            int32_t hids[32*8];
            for (int t = 0; t < (int)pNy; ++t) {
                cudaMemcpyAsync(hids + (size_t)t*nids,
                                (const char *)pids->data + (size_t)t*pids->nb[1],
                                nids*sizeof(int32_t), cudaMemcpyDeviceToHost, ctx.stream());
            }
            cudaStreamSynchronize(ctx.stream());
            bool seen[1024]; for (int z = 0; z < 1024; ++z) seen[z] = false;
            int uni = 0;
            for (int k = 1; k <= (int)pNy; ++k) {
                for (int j = 0; j < nids; ++j) {
                    int e = hids[(size_t)(k-1)*nids + j];
                    if (e >= 0 && e < 1024 && !seen[e]) { seen[e] = true; ++uni; }
                }
                e2_union[k] += uni; e2_cnt[k] += 1;
            }
            if (++e2_calls % 100 == 0) {
                fprintf(stderr, "PXA_E2PXQ calls=%d nids=%d dev=%d union/k:", e2_calls, nids, ctx.device);
                for (int k = 1; k <= 8; ++k) if (e2_cnt[k])
                    fprintf(stderr, " k%d=%.2f/%d", k, e2_union[k]/e2_cnt[k], k*nids);
                fprintf(stderr, "\n");
            }
        }
    }
    const int64_t n_ids = ids->ne[0];
    const int64_t Ny    = src1->ne[2];
    const int   unary = (uop == GGML_UNARY_OP_SWIGLU_OAI) ? 1 : 0;
    const float limit = (uop == GGML_UNARY_OP_SWIGLU_OAI) ? 7.0f : *(const float *)(dst->op_params + 1);

    const ggml_tensor * bu = dst->src[4], * bg = dst->src[5];

    const int  pair = pxa_pxq6_decode_mode();   // sourcing MODE (tab/pairlut/prmt x cs) — name kept for call-site diff-min
    const bool vecx = pxa_pxq6_vecx();
    const int  sgen = pxa_pxq6_ksplit_gen_eff(ggml_cuda_info().devices[ctx.device].cc);
    const bool newfam = fmt >= PXA_PXQ_FMT_P6 || fmt_g >= PXA_PXQ_FMT_P6 || pair || vecx || sgen || pxa_pxq6_ksplit();

    dim3 grid((unsigned)(R/PXQ4_BM), (unsigned)n_ids, (unsigned)Ny);

    // G2-F1 REDFUSE eligibility (PXA_G2_REDFUSE=1, bit-exact KSPLIT form only): the down
    // MUL_MAT_ID immediately follows consuming EXACTLY this gateup dst (same ids), the gateup
    // dst has NO other consumer anywhere in the graph, and a redfuse kernel exists for the down
    // format. When it fires, the standalone k_pxq6_gateup_reduce is skipped and the down mmv
    // reconstructs its x from the KSPLIT partials (bit-identical by construction).
    bool redfuse_ok = false;
    int  rf_fmt_d   = PXA_PXQ_FMT_NONE;
    if (pxa_g2_redfuse() && !sgen && pxa_pxq6_ksplit() && graph &&
        next && next->op == GGML_OP_MUL_MAT_ID &&
        next->src[1] == dst && next->src[2] == ids &&
        next->src[0]->ne[0] == dst->ne[0] && next->src[0]->ne[0] % PXQ4_QK == 0 &&
        next->src[0]->ne[1] % PXQ4_BM == 0 &&
        pxa_pxq_fmt(next->src[0]->type) != PXA_PXQ_FMT_NONE &&
        pxa_pxq4_bufs_on_device(ctx, {next->src[0], next})) {
        rf_fmt_d = pxa_pxq_fmt(next->src[0]->type);
        const int64_t Kd_rf = next->src[0]->ne[0];
        const size_t smem_d_rf = (size_t)Kd_rf*sizeof(float) + PXQ4_MMV_KSEG*64*sizeof(float);
        if (smem_d_rf <= 46*1024 && pxq6_pick_mmv_redfuse(rf_fmt_d, pair, vecx)) {
            bool sole = true;   // sole-consumer guard: nothing besides `next` may reference dst
            for (int jn = i + 2; jn < graph->n_nodes && sole; ++jn) {
                const ggml_tensor * nn = graph->nodes[jn];
                if (nn->view_src == dst) { sole = false; break; }
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    if (nn->src[s] == dst) { sole = false; break; }
                }
            }
            redfuse_ok = sole;
        }
    }

    // PXQ_CANON_v1: clamp the gen split S to the canonical chunk count so it always divides
    // NFIX (bit-exactness precondition); an S clamped below 2 falls back to the K1/unsplit path.
    const int kslabs_gu = (int)(K/PXQ4_QK);
    const int nfix_gu   = pxq6_canon_nfix(kslabs_gu, PXQ6_GU_SPLIT_MAX);
    const int sgen_eff  = sgen > nfix_gu ? nfix_gu : sgen;

    bool gu_done = false;
    if (newfam && (pxa_pxq6_ksplit() || sgen_eff >= 2)) {
        // K1: split gateup launch + fixed-order reducer. Workspace grows only outside graph
        // capture; if unavailable we fall through to the unsplit kernel (bit-identical for the
        // bit-exact form, so replayed graphs stay correct whatever capture saw).
        const size_t need = (size_t)Ny*n_ids*2*(sgen_eff >= 2 ? PXQ6_GU_SPLIT_MAX*PXQ4_MMV_KSEG : PXQ4_MMV_KSEG)*R;
        float * ws = pxq6_ksplit_workspace(ctx.device, stream, need);
        if (ws) {
            const int rblk = pxa_pxq_reduce_blk();
                        dim3 gridr((unsigned)((R + rblk - 1)/rblk), (unsigned)n_ids, (unsigned)Ny);
            if (sgen_eff >= 2) {
                const int kslabs = kslabs_gu;
                const int kcmax  = ((kslabs + sgen_eff - 1)/sgen_eff)*PXQ4_QK;
                const size_t smem_s = (size_t)kcmax*sizeof(float);
                // PXA_PXQ_TOKBATCH (2026-08-04): TOK-batched verify. On an Ny>1 verify
                // batch, fold TOK consecutive tokens into one block (grid.z = ceil(Ny/TOK))
                // so the expert weight walk is shared across the group's tokens (kernel
                // header in pxq6.cuh). ws sizing and the canonical reducer are IDENTICAL to
                // the per-token arm -- output is bit-identical; only the launch shape and
                // the weight-read temporal locality change. Declines (lever off, Ny==1, no
                // instantiation, smem cap) fall through to the per-token launch unchanged.
                const int tok_req = pxa_pxq_tokbatch();
                if (tok_req >= 2 && Ny >= 2) {
                    const int tok = Ny < (int64_t)tok_req ? (int)Ny : tok_req;
                    auto * kst = pxq6_pick_gateup_ksplit_gen_tok(fmt, fmt_g, pair, vecx, tok);
                    const size_t smem_t = (size_t)tok*kcmax*sizeof(float);
                    if (kst && smem_t <= 46*1024) {
                        dim3 gridt((unsigned)(R/PXQ4_BM*sgen_eff), (unsigned)n_ids,
                                   (unsigned)((Ny + tok - 1)/tok));
                        pxa_pxq_tokbatch_log(ctx.device, true, tok, gridt.x, gridt.y, gridt.z,
                                             (int)Ny, sgen_eff, smem_t, fmt, fmt_g, pair, (int)vecx);
                        kst<<<gridt, 256, smem_t, stream>>>(
                            (const uint8_t *)src0_1->data, (const uint8_t *)src0_2->data,
                            (const char *)src1->data, src1->nb[2], ws,
                            (const char *)ids->data, ids->nb[0], ids->nb[1],
                            (int)R, (int)K, (int)n_as, (int)n_ids, sgen_eff, (int)Ny);
                        CUDA_CHECK(cudaGetLastError());
                        k_pxq6_gateup_reduce_gen<<<gridr, rblk, 0, stream>>>(ws,
                            (char *)dst->data, dst->nb[2], dst->nb[1],
                            (const char *)ids->data, ids->nb[0], ids->nb[1],
                            bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0,
                            bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0,
                            (int)R, (int)n_as, (int)n_ids, unary, 1.702f, limit, kslabs);
                        CUDA_CHECK(cudaGetLastError());
                        gu_done = true;
                    } else {
                        pxa_pxq_tokbatch_log(ctx.device, false, tok, 0, 0, 0,
                                             (int)Ny, sgen_eff, smem_t, fmt, fmt_g, pair, (int)vecx);
                    }
                }
                dim3 grids((unsigned)(R/PXQ4_BM*sgen_eff), (unsigned)n_ids, (unsigned)Ny);
                auto * ks = pxq6_pick_gateup_ksplit_gen(fmt, fmt_g, pair, vecx);
                int nthr_gug = 256;
                if (ks && pxa_pxq6_rowx2()) {
                    auto * k2 = pxq6_pick_gateup_ksplit_gen_x2(fmt, fmt_g, pair, vecx);
                    pxa_pxq6_rowx2_log(2, k2 != nullptr);
                    if (k2) { ks = k2; nthr_gug = 128; }
                }
                if (ks && nthr_gug == 256 && (pxa_pxq6_qpf() & 2)) {   // Q1 yields to an engaged R2 twin
                    auto * kq = pxq6_pick_gateup_ksplit_gen_qp(fmt, fmt_g, pair, vecx);
                    pxa_pxq6_qpf_log(2, kq != nullptr);
                    if (kq) ks = kq;
                }
                if (!gu_done && ks) {
                int * ctrs = pxa_pxq_split_fusered()
                    ? pxq6_ksplit_counters(ctx.device, stream) : nullptr;
                ks<<<grids, nthr_gug, smem_s, stream>>>(
                    (const uint8_t *)src0_1->data, (const uint8_t *)src0_2->data,
                    (const char *)src1->data, src1->nb[2], ws,
                    (const char *)ids->data, ids->nb[0], ids->nb[1],
                    (char *)dst->data, dst->nb[2], dst->nb[1],
                    bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0,
                    bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0,
                    ctrs, unary, 1.702f, limit,
                    (int)R, (int)K, (int)n_as, (int)n_ids, sgen_eff);
                CUDA_CHECK(cudaGetLastError());
                if (!ctrs) {
                    k_pxq6_gateup_reduce_gen<<<gridr, rblk, 0, stream>>>(ws,
                        (char *)dst->data, dst->nb[2], dst->nb[1],
                        (const char *)ids->data, ids->nb[0], ids->nb[1],
                        bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0,
                        bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0,
                        (int)R, (int)n_as, (int)n_ids, unary, 1.702f, limit, kslabs);
                    CUDA_CHECK(cudaGetLastError());
                }
                gu_done = true;
                }
            } else {
                dim3 grids((unsigned)(R/PXQ4_BM*PXQ4_MMV_KSEG), (unsigned)n_ids, (unsigned)Ny);
                auto * ks = pxq6_pick_gateup_ksplit(fmt, fmt_g, pair, vecx);
                if (ks) {
                ks<<<grids, 64, (size_t)K*sizeof(float), stream>>>(
                    (const uint8_t *)src0_1->data, (const uint8_t *)src0_2->data,
                    (const char *)src1->data, src1->nb[2], ws,
                    (const char *)ids->data, ids->nb[0], ids->nb[1],
                    (int)R, (int)K, (int)n_as, (int)n_ids);
                CUDA_CHECK(cudaGetLastError());
                if (redfuse_ok) {
                    // G2-F1: skip the standalone reduce; the down mmv reconstructs x from ws.
                    const int64_t Rd = next->src[0]->ne[1], Kd = next->src[0]->ne[0];
                    const size_t smem_d = (size_t)Kd*sizeof(float) + PXQ4_MMV_KSEG*64*sizeof(float);
                    dim3 gridd((unsigned)(Rd/PXQ4_BM), (unsigned)n_ids, (unsigned)Ny);
                    auto * kern_rf = pxq6_pick_mmv_redfuse(rf_fmt_d, pair, vecx);
                    kern_rf<<<gridd, 256, smem_d, stream>>>(
                        (const uint8_t *)next->src[0]->data, ws,
                        (char *)next->data, next->nb[2], next->nb[1],
                        (const char *)ids->data, ids->nb[0], ids->nb[1],
                        bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0,
                        bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0,
                        (int)Rd, (int)Kd, (int)n_as, (int)n_ids, unary, 1.702f, limit);
                    CUDA_CHECK(cudaGetLastError());
                    return i + 1;
                }
                k_pxq6_gateup_reduce<<<gridr, rblk, 0, stream>>>(ws,
                    (char *)dst->data, dst->nb[2], dst->nb[1],
                    (const char *)ids->data, ids->nb[0], ids->nb[1],
                    bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0,
                    bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0,
                    (int)R, (int)n_as, (int)n_ids, unary, 1.702f, limit);
                CUDA_CHECK(cudaGetLastError());
                gu_done = true;
                }
            }
        }
    }
    if (!gu_done) {
        // every remaining PXA slab format is new-family (the id-250/251 legacy kernels were
        // removed 2026-07-21); newfam is true by construction here.
        auto * kern_gu = pxq6_pick_gateup(fmt, fmt_g, pair, vecx);
        if (!kern_gu) return -1;
        kern_gu<<<grid, 256, smem_gu, stream>>>(
            (const uint8_t *)src0_1->data, (const uint8_t *)src0_2->data,
            (const char *)src1->data, src1->nb[2],
            (char *)dst->data, dst->nb[2], dst->nb[1],
            (const char *)ids->data, ids->nb[0], ids->nb[1],
            bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0,
            bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0,
            (int)R, (int)K, (int)n_as, unary, 1.702f, limit);
        CUDA_CHECK(cudaGetLastError());
    }

    // fuse the PXQ4 down-proj when it is the next node (mirrors the stock fast-TG fusion;
    // a following ADD_ID runs as its own node)
    if (next && next->op == GGML_OP_MUL_MAT_ID &&
        next->src[1] == dst && next->src[2] == ids &&
        next->src[0]->ne[0] == dst->ne[0] && next->src[0]->ne[0] % PXQ4_QK == 0 &&
        next->src[0]->ne[1] % PXQ4_BM == 0 &&
        pxa_pxq_fmt(next->src[0]->type) != PXA_PXQ_FMT_NONE &&
        pxa_pxq4_bufs_on_device(ctx, {next->src[0], next})) {
        const int fmt_d = pxa_pxq_fmt(next->src[0]->type);
        const int64_t Rd = next->src[0]->ne[1], Kd = next->src[0]->ne[0];
        const size_t smem_d = (size_t)Kd*sizeof(float) + PXQ4_MMV_KSEG*64*sizeof(float);
        if (smem_d <= 46*1024) {
            // MoE down split (PXA_PXQ_MOE_DOWN_SPLIT, default OFF -- rationale + the ws-tax
            // arithmetic live on the lever above). Bit-exact both arms; any decline (lever
            // off, no instantiation, ws unavailable under graph capture) falls through to the
            // unsplit launch below, which produces the same bytes.
            const int sdn_req = pxa_pxq_moe_down_split();
            if (sdn_req == 1) {
                // K1 kseg split: 64-thr blocks, one of the KSEG lanes each; per-thread fp32
                // chain byte-identical to k_pxq6_mmv's; k_pxq_mmv_reduce replays the red[]
                // combination order. ws = KSEG*R floats per (iy, slot).
                auto * ks_d = pxq6_pick_mmv_ksplit(fmt_d, pair, vecx);
                float * ws_d = nullptr;
                if (ks_d) {
                    const size_t need_d = (size_t)Ny*n_ids*PXQ4_MMV_KSEG*(size_t)Rd;
                    ws_d = pxq6_ksplit_workspace(ctx.device, stream, need_d);
                }
                pxa_pxq_moe_down_split_log(ctx.device, ws_d != nullptr, 1,
                        (int)(Rd/PXQ4_BM)*PXQ4_MMV_KSEG, (int)n_ids, (int)Ny, (int)Rd, (int)Kd,
                        pxq6_canon_nfix((int)(Kd/PXQ6_QK), PXQ6_MMV_SPLIT_MAX));
                if (ws_d) {
                    const size_t smem_ks = (size_t)Kd*sizeof(float);   // no red[] in the 64-thr form
                    dim3 gridsd((unsigned)((Rd/PXQ4_BM)*PXQ4_MMV_KSEG), (unsigned)n_ids, (unsigned)Ny);
                    ks_d<<<gridsd, 64, smem_ks, stream>>>(
                        (const uint8_t *)next->src[0]->data,
                        (const char *)dst->data, dst->nb[2], dst->nb[1],
                        ws_d,
                        (const char *)ids->data, ids->nb[0], ids->nb[1],
                        (int)Rd, (int)Kd, (int)n_as, (int)n_ids);
                    CUDA_CHECK(cudaGetLastError());
                    const int rblk = pxa_pxq_reduce_blk();
                    dim3 gridrd((unsigned)((Rd + rblk - 1)/rblk), (unsigned)n_ids, (unsigned)Ny);
                    k_pxq_mmv_reduce<<<gridrd, rblk, 0, stream>>>(ws_d,
                        (char *)next->data, next->nb[2], next->nb[1],
                        (const char *)ids->data, ids->nb[0], ids->nb[1],
                        (int)Rd, (int)n_as, (int)n_ids);
                    CUDA_CHECK(cudaGetLastError());
                    return i + 1;
                }
            } else if (sdn_req >= 2) {
                // gen S-split: mirrors the K8-2D dispatch (ggml_cuda.cu K8-2D block) with the
                // REAL ids strides. S clamped to divide nfix (both powers of two, S <= nfix
                // => divides) so split == unsplit bit-for-bit; ws sized on the MMV cap (see
                // the lever comment -- the GU cap would undersize the row stride 2x).
                const int kslabs_d = (int)(Kd/PXQ6_QK);
                const int nfix_d   = pxq6_canon_nfix(kslabs_d, PXQ6_MMV_SPLIT_MAX);
                const int sdn      = sdn_req > nfix_d ? nfix_d : sdn_req;
                auto * ksg_d = sdn >= 2 ? pxq6_pick_mmv_ksplit_gen(fmt_d, pair, vecx) : nullptr;
                int nthr_dks = 256;
                if (ksg_d && pxa_pxq6_rowx2()) {
                    auto * k2 = pxq6_pick_mmv_ksplit_gen_x2(fmt_d, pair, vecx);
                    pxa_pxq6_rowx2_log(4, k2 != nullptr);
                    if (k2) { ksg_d = k2; nthr_dks = 128; }
                }
                if (ksg_d && nthr_dks == 256 && (pxa_pxq6_qpf() & 1)) {   // Q1 yields to an engaged R2 twin
                    auto * kq = pxq6_pick_mmv_ksplit_gen_qp(fmt_d, pair, vecx);
                    pxa_pxq6_qpf_log(4, kq != nullptr);
                    if (kq) ksg_d = kq;
                }
                float * ws_d = nullptr;
                if (ksg_d) {
                    const size_t need_d = (size_t)Ny*n_ids*(size_t)(PXQ6_MMV_SPLIT_MAX*PXQ4_MMV_KSEG)*(size_t)Rd;
                    ws_d = pxq6_ksplit_workspace(ctx.device, stream, need_d);
                }
                pxa_pxq_moe_down_split_log(ctx.device, ws_d != nullptr, sdn,
                        (int)(Rd/PXQ4_BM)*sdn, (int)n_ids, (int)Ny, (int)Rd, (int)Kd, nfix_d);
                if (ws_d) {
                    int * ctrs_d = pxa_pxq_split_fusered()
                        ? pxq6_ksplit_counters(ctx.device, stream) : nullptr;
                    const int kc_max_d = (kslabs_d + sdn - 1)/sdn;
                    const size_t smem_sd = (size_t)kc_max_d*PXQ6_QK*sizeof(float);
                    dim3 gridsd((unsigned)((Rd/PXQ4_BM)*sdn), (unsigned)n_ids, (unsigned)Ny);
                    ksg_d<<<gridsd, nthr_dks, smem_sd, stream>>>(
                        (const uint8_t *)next->src[0]->data,
                        (const char *)dst->data, dst->nb[2], dst->nb[1],
                        ws_d,
                        (const char *)ids->data, ids->nb[0], ids->nb[1],
                        (char *)next->data, next->nb[2], next->nb[1], ctrs_d,
                        (int)Rd, (int)Kd, (int)n_as, (int)n_ids, sdn);
                    CUDA_CHECK(cudaGetLastError());
                    if (!ctrs_d) {
                        const int rblk = pxa_pxq_reduce_blk();
                        dim3 gridrd((unsigned)((Rd + rblk - 1)/rblk), (unsigned)n_ids, (unsigned)Ny);
                        k_pxq_mmv_reduce_s<<<gridrd, rblk, 0, stream>>>(ws_d,
                            (char *)next->data, next->nb[2], next->nb[1],
                            (const char *)ids->data, ids->nb[0], ids->nb[1],
                            (int)Rd, (int)n_as, (int)n_ids, kslabs_d);
                        CUDA_CHECK(cudaGetLastError());
                    }
                    return i + 1;
                }
            }
            dim3 gridd((unsigned)(Rd/PXQ4_BM), (unsigned)n_ids, (unsigned)Ny);
            auto * kern_d = pxq6_pick_mmv(fmt_d, pair, vecx);
            if (!kern_d) return i;   // no kernel for this format combination — skip the fusion
            int nthr_dmm = 256;
            if (pxa_pxq6_rowx2()) {
                auto * k2 = pxq6_pick_mmv_x2(fmt_d, pair, vecx);
                pxa_pxq6_rowx2_log(5, k2 != nullptr);
                if (k2) { kern_d = k2; nthr_dmm = 128; }
            }
            if (nthr_dmm == 256 && (pxa_pxq6_qpf() & 1)) {   // Q1 yields to an engaged R2 twin
                auto * kq = pxq6_pick_mmv_qp(fmt_d, pair, vecx);
                pxa_pxq6_qpf_log(5, kq != nullptr);
                if (kq) kern_d = kq;
            }
            kern_d<<<gridd, nthr_dmm, smem_d, stream>>>(
                (const uint8_t *)next->src[0]->data,
                (const char *)dst->data, dst->nb[2], dst->nb[1],
                (char *)next->data, next->nb[2], next->nb[1],
                (const char *)ids->data, ids->nb[0], ids->nb[1],
                (int)Rd, (int)Kd, (int)n_as);
            CUDA_CHECK(cudaGetLastError());
            return i + 1;
        }
    }
    return i;
}

// prefill: grouped fused GEMMs over ALL routed experts in one launch per projection
static int pxa_pxq4_moe_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cgraph * graph, int i) {
    ggml_tensor * next = i + 1 < graph->n_nodes ? graph->nodes[i+1] : nullptr;
    const ggml_tensor * src0_1 = dst->src[0];   // up   (bias = dst->src[4])
    const ggml_tensor * src0_2 = dst->src[1];   // gate (bias = dst->src[5])
    const ggml_tensor * src1   = dst->src[2];
    const ggml_tensor * ids    = dst->src[3];

    const int fmt   = pxa_pxq_fmt(src0_1->type);                       // up
    const int fmt_g = src0_2 ? pxa_pxq_fmt(src0_2->type) : PXA_PXQ_FMT_NONE;  // gate
    if (!src0_2 || fmt == PXA_PXQ_FMT_NONE || fmt_g == PXA_PXQ_FMT_NONE) return -1;
    pxa_pxq_dispatch_note("moe_gateup_prefill", fmt, fmt_g);
    if (fmt >= PXA_PXQ_FMT_P6 || fmt_g >= PXA_PXQ_FMT_P6) pxq6_maybe_upload_tables(ctx.device);
    if (fmt >= PXA_PXQ_FMT_P2 || fmt_g >= PXA_PXQ_FMT_P2) pxq23_maybe_upload_books(ctx.device);
    const ggml_unary_op uop = (ggml_unary_op)dst->op_params[0];
    if (uop != GGML_UNARY_OP_SWIGLU_OAI && uop != GGML_UNARY_OP_SILU) return -1;
    if (src1->type != GGML_TYPE_F32 || src1->ne[1] != 1 || src1->ne[3] != 1) return -1;
    if (!ggml_is_contiguous(dst)) return -1;
    const int64_t R = src0_1->ne[1], K = src0_1->ne[0];
    if (R % PXQ4_BM || K % PXQ4_QK || src0_2->ne[0] != K || src0_2->ne[1] != R) return -1;
    if (!pxa_pxq4_bufs_on_device(ctx, {src0_1, src0_2, src1, dst})) return -1;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!(fast_fp16_available(cc) && cc < CC_TURING) && !pxa_pxq6_force_prefill()) return -1;   // half2 kernel: sm_60 / sm_70

    cudaStream_t stream = ctx.stream();
    const int64_t n_as  = src0_1->ne[2];
    const int64_t n_ids = ids->ne[0];
    const int   unary = (uop == GGML_UNARY_OP_SWIGLU_OAI) ? 1 : 0;
    const float glu_limit = (uop == GGML_UNARY_OP_SWIGLU_OAI) ? 7.0f : *(const float *)(dst->op_params + 1);

    const bool fuse_down = next && next->op == GGML_OP_MUL_MAT_ID &&
        pxa_pxq_fmt(next->src[0]->type) != PXA_PXQ_FMT_NONE && next->src[1] == dst && next->src[2] == ids &&
        next->src[0]->ne[0] == dst->ne[0] && next->src[0]->ne[0] % PXQ4_QK == 0 &&
        next->src[0]->ne[1] % PXQ4_BM == 0 && ggml_is_contiguous(next) &&
        pxa_pxq4_bufs_on_device(ctx, {next->src[0], next});

    // Row mapping + tile list. PXA_MOE_DEVICE_MAP=1 builds both on the device (no ids readback,
    // no cudaStreamSynchronize inside graph compute); the grids are then sized for the worst
    // case and the padding tiles / mapping slots are skipped by the consuming kernels.
    ggml_cuda_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool());
    ggml_cuda_pool_alloc<pxq4_tile_info>   dev_tiles(ctx.pool());
    pxq_moemap_bufs moemap(ctx.pool());
    const int  n_rows_max = (int)(ids->ne[1]*n_ids);
    const int  n_tiles_wc = pxq_moemap_tiles_max(n_rows_max, (int)n_as);
    const bool devmap     = pxa_moe_device_map() && n_rows_max > 0 &&
                            n_as <= PXQ_MOEMAP_MAXAS && n_tiles_wc <= 65535;
    int64_t total;
    size_t  n_tiles;

    if (devmap) {
        total   = n_rows_max;                 // == the routed row count unless an id is out of range
        n_tiles = (size_t)n_tiles_wc;
        dev_row_mapping.alloc((size_t)total);
        dev_tiles.alloc(n_tiles);
        if (!pxq_moemap_build(ctx, ids, n_as, n_ids, n_rows_max, n_tiles_wc, moemap,
                              (pxq4_rowmap *)dev_row_mapping.get(), dev_tiles.get())) return -1;
        if (pxa_moe_device_map_mode() >= 2) {
            pxq_moemap_verify(ctx, ids, n_as, n_ids, n_rows_max, n_tiles_wc,
                              (const pxq4_rowmap *)dev_row_mapping.get(), dev_tiles.get());
        }
        // stands in for the host path's is_ser memset (see pxq-moemap.cuh)
        pxq_moemap_zero_unrouted(ctx, ids, fuse_down ? next : dst, n_as, n_ids, n_rows_max);
    } else {
        std::vector<int> moe_counts, cum_moe_counts;
        bool is_ser = prepare_row_mappigs(ctx, n_as, n_ids, ids, moe_counts, cum_moe_counts, dev_row_mapping);
        if (is_ser) {
            ggml_tensor * t = fuse_down ? next : dst;
            CUDA_CHECK(cudaMemsetAsync(t->data, 0, ggml_nbytes(t), stream));
        }
        total = cum_moe_counts[n_as];
        if (total == 0) return fuse_down ? i + 1 : i;

        // host tile map: (expert, flat row0, nrows<=64) per 64-token tile, expert-grouped order
        std::vector<pxq4_tile_info> tiles;
        tiles.reserve((size_t)(total/PXQ4_BN + n_as + 1));
        for (int e = 0; e < (int)n_as; ++e) {
            for (int t0 = 0; t0 < moe_counts[e]; t0 += PXQ4_BN) {
                tiles.push_back({e, cum_moe_counts[e] + t0, std::min((int)PXQ4_BN, moe_counts[e] - t0), 0});
            }
        }
        if (tiles.empty() || tiles.size() > 65535) return -1;   // grid.y limit; fallback handles it
        n_tiles = tiles.size();
        dev_tiles.alloc(n_tiles);
        CUDA_CHECK(cudaMemcpyAsync(dev_tiles.get(), tiles.data(), n_tiles*sizeof(pxq4_tile_info),
                                   cudaMemcpyHostToDevice, stream));
    }

    // gather activations once (f32 -> f16, same convert the cublas fp16 path does)
    ggml_cuda_pool_alloc<half> A_f16(ctx.pool(), total*K);
    k_pxq4_gather_a_f16<<<(unsigned)total, 256, 0, stream>>>((const char *)src1->data, A_f16.get(),
            (const pxq4_rowmap *)dev_row_mapping.get(), K, src1->ne[1], src1->nb[1], src1->nb[2]);
    CUDA_CHECK(cudaGetLastError());

    // up + gate grouped GEMMs (biases folded into the epilogue).
    // Kernel-family selection (PXQ6 spec K3/K4/K5/K6): with NO PXA_PXQ6_* prefill gate set,
    // (The legacy id-250/251 old-family prefill kernels were removed 2026-07-21.)
    const ggml_tensor * bu = dst->src[4], * bg = dst->src[5];
    dim3 grid((unsigned)(R/PXQ4_BM), (unsigned)n_tiles);

    const int  wmma_env  = cc == 700 ? pxa_pxq6_wmma() : 0;   // K6: exactly Volta (sm_70)
    // v1 (modes 1/2) excludes 2/3-bit + mixed pairs; v2 (mode 3) covers P2/P3/P6/P6HQ + mixed
    // pairs but excludes P6R and requires GUFUSE (its fused kernel IS the up+gate path).
    // P1 is excluded from K6 v2 for the same reason as P6R: PXQ6_WMMA_GU_PICK has no arm for
    // it, and both gufuse call sites launch the picked pointer WITHOUT a null check. P1 keeps
    // the (policy-generic) non-WMMA K5 gufuse path, which does have a P1 arm.
    const int  wmma_mode = wmma_env == 3
        ? ((fmt != PXA_PXQ_FMT_P6R && fmt_g != PXA_PXQ_FMT_P6R &&
            fmt != PXA_PXQ_FMT_P1  && fmt_g != PXA_PXQ_FMT_P1  && pxa_pxq6_gufuse()) ? 3 : 0)
        : ((fmt < PXA_PXQ_FMT_P2 && fmt == fmt_g) ? wmma_env : 0);
    const bool wmma2     = wmma_mode == 3;   // K6 v2: double-buffered, GUFUSE-fused, WMMA scat down
    const bool rag    = pxa_pxq6_ragtail();
    const bool pipe   = pxa_pxq6_pipe();
    const bool newfam = fmt >= PXA_PXQ_FMT_P6 || fmt_g >= PXA_PXQ_FMT_P6 || wmma_mode || rag || pipe ||
                        pxa_pxq6_gufuse() || pxa_pxq6_scatfuse() || pxa_pxq6_force_prefill();
    if (!newfam) return -1;   // belt: every remaining PXA slab format is new-family
    const bool gufuse = newfam && pxa_pxq6_gufuse() && (!wmma_mode || wmma2) && (fmt == fmt_g || wmma2);   // K6 v1 keeps split GEMMs; v2 has its own fused (mixed-pair-capable) kernel
    const bool scat   = newfam && pxa_pxq6_scatfuse();
    // K6 launch fix (2026-07-19): k_pxq6_gemm_grouped_wmma is a 256-thread (8-warp) block
    // (__launch_bounds__(256); srow = tid>>2 stages 64 rows x 4 thr; wm/wn = 4x2 warp grid).
    // It was being launched with the plain grouped gemm's 64 threads -> 3/4 of the smem tile
    // unstaged + 6/8 warps missing = garbage output (no CUDA error: launch_bounds is a max).
    const unsigned nthr_gu = wmma_mode ? 256u : 64u;

    pxq6_gemm_fn kern_up   = (wmma_mode && !wmma2) ? pxq6_pick_gemm_wmma(fmt,   wmma_mode != 2) : pxq6_pick_gemm(fmt,   rag, pipe);
    pxq6_gemm_fn kern_gate = (wmma_mode && !wmma2) ? pxq6_pick_gemm_wmma(fmt_g, wmma_mode != 2) : pxq6_pick_gemm(fmt_g, rag, pipe);
    if (!kern_up || !kern_gate) return -1;

    if (fuse_down) {
        const int64_t Rd = next->src[0]->ne[1], Kd = next->src[0]->ne[0];   // Kd == R
        const int fmt_d = pxa_pxq_fmt(next->src[0]->type);
            ggml_cuda_pool_alloc<half> H_f16(ctx.pool(), total*R);
        if (gufuse) {
            // K3 GUFUSE: one kernel = up GEMM + gate GEMM + GLU -> f16 (bit-exact vs the
            // 3-kernel pipeline; identical accumulation chains + identical GLU/convert)
            auto * kg = wmma2 ? pxq6_pick_gufuse_wmma_h(fmt, fmt_g) : pxq6_pick_gufuse_h(fmt, rag, pipe);
            kg<<<grid, wmma2 ? 256u : 64u, 0, stream>>>((const uint8_t *)src0_1->data, (const uint8_t *)src0_2->data,
                    A_f16.get(), H_f16.get(),
                    bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0,
                    bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0,
                    dev_tiles.get(), (int)R, (int)K, unary, 1.702f, glu_limit);
            CUDA_CHECK(cudaGetLastError());
        } else {
            ggml_cuda_pool_alloc<float> C_up(ctx.pool(), total*R);
            ggml_cuda_pool_alloc<float> C_gate(ctx.pool(), total*R);
            kern_up<<<grid, nthr_gu, 0, stream>>>((const uint8_t *)src0_1->data, A_f16.get(), C_up.get(),
                    bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0, dev_tiles.get(), (int)R, (int)K);
            kern_gate<<<grid, nthr_gu, 0, stream>>>((const uint8_t *)src0_2->data, A_f16.get(), C_gate.get(),
                    bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0, dev_tiles.get(), (int)R, (int)K);
            CUDA_CHECK(cudaGetLastError());
            const int64_t k = total*R;
            k_pxq4_glu<half><<<(unsigned)((k + 255)/256), 256, 0, stream>>>(
                    C_gate.get(), C_up.get(), H_f16.get(), k, unary, 1.702f, glu_limit);
        }
        dim3 gridd((unsigned)(Rd/PXQ4_BM), (unsigned)n_tiles);
        if (scat) {
            // K3 SCATFUSE: down GEMM scatters straight to the MoE output rows
            pxq6_scat_fn kd = wmma2 && fmt_d != PXA_PXQ_FMT_P6R ? pxq6_pick_down_scat_wmma(fmt_d) : nullptr;
            const unsigned nthr_d = kd ? 256u : 64u;
            if (!kd) kd = pxq6_pick_down_scat(fmt_d, rag, pipe);
            kd<<<gridd, nthr_d, 0, stream>>>((const uint8_t *)next->src[0]->data, H_f16.get(),
                    (char *)next->data, next->nb[1], next->nb[2],
                    (const pxq4_rowmap *)dev_row_mapping.get(), dev_tiles.get(), (int)Rd, (int)Kd);
            CUDA_CHECK(cudaGetLastError());
        } else {
            ggml_cuda_pool_alloc<float> C_down(ctx.pool(), total*Rd);
            const bool down_wmma = wmma_mode && !wmma2 && fmt_d == fmt;
            pxq6_gemm_fn kern_down = down_wmma ? pxq6_pick_gemm_wmma(fmt_d, wmma_mode != 2) : pxq6_pick_gemm(fmt_d, rag, pipe);
            kern_down<<<gridd, down_wmma ? 256u : 64u, 0, stream>>>((const uint8_t *)next->src[0]->data, H_f16.get(),
                    C_down.get(), nullptr, 0, dev_tiles.get(), (int)Rd, (int)Kd);
            CUDA_CHECK(cudaGetLastError());
            dim3 bd(std::min((unsigned)Rd, 768u));
            k_copy_dst_from_contiguous<<<(unsigned)total, bd, 0, stream>>>((char *)next->data,
                    (const char *)C_down.get(), dev_row_mapping.get(), Rd, next->nb[1], next->nb[2]);
            CUDA_CHECK(cudaGetLastError());
        }
        return i + 1;
    }

    // no down fusion: GLU to f32, scatter to dst
    ggml_cuda_pool_alloc<float> Out(ctx.pool(), total*R);
    if (gufuse) {
        auto * kg = wmma2 ? pxq6_pick_gufuse_wmma_f(fmt, fmt_g) : pxq6_pick_gufuse_f(fmt, rag, pipe);
        kg<<<grid, wmma2 ? 256u : 64u, 0, stream>>>((const uint8_t *)src0_1->data, (const uint8_t *)src0_2->data,
                A_f16.get(), Out.get(),
                bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0,
                bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0,
                dev_tiles.get(), (int)R, (int)K, unary, 1.702f, glu_limit);
        CUDA_CHECK(cudaGetLastError());
    } else {
        ggml_cuda_pool_alloc<float> C_up(ctx.pool(), total*R);
        ggml_cuda_pool_alloc<float> C_gate(ctx.pool(), total*R);
        kern_up<<<grid, nthr_gu, 0, stream>>>((const uint8_t *)src0_1->data, A_f16.get(), C_up.get(),
                bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0, dev_tiles.get(), (int)R, (int)K);
        kern_gate<<<grid, nthr_gu, 0, stream>>>((const uint8_t *)src0_2->data, A_f16.get(), C_gate.get(),
                bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0, dev_tiles.get(), (int)R, (int)K);
        CUDA_CHECK(cudaGetLastError());
        const int64_t k = total*R;
        k_pxq4_glu<float><<<(unsigned)((k + 255)/256), 256, 0, stream>>>(
                C_gate.get(), C_up.get(), Out.get(), k, unary, 1.702f, glu_limit);
    }
    dim3 bd(std::min((unsigned)R, 768u));
    k_copy_dst_from_contiguous<<<(unsigned)total, bd, 0, stream>>>((char *)dst->data,
            (const char *)Out.get(), dev_row_mapping.get(), R, dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
    return i;
}

// N13: int8 dp4a MMQ-tile prefill (env PXA_PXQ_INT8_PREFILL, default OFF; see pxq6i8.cuh for
// the design + numeric contract). Mirrors pxa_pxq4_moe_prefill's setup so the proven driver
// stays textually untouched; every decline (-1) falls through to it. NOT bit-exact vs the
// fused fp16 pipeline (G3-gated). Decode never reaches here (fast-TG owns Ny<=8), so decode
// is byte-identical whatever the flag.
static int pxa_pxq_moe_prefill_i8(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cgraph * graph, int i) {
    const int i8mode = pxa_pxq_int8_prefill();
    if (i8mode == 0) return -1;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (i8mode == 1 && cc != 610) return -1;   // ship gate: exactly sm_61; mode 2 = any arch (TEST)

    ggml_tensor * next = i + 1 < graph->n_nodes ? graph->nodes[i+1] : nullptr;
    const ggml_tensor * src0_1 = dst->src[0];   // up   (bias = dst->src[4])
    const ggml_tensor * src0_2 = dst->src[1];   // gate (bias = dst->src[5])
    const ggml_tensor * src1   = dst->src[2];
    const ggml_tensor * ids    = dst->src[3];

    const int fmt   = src0_1 ? pxa_pxq_fmt(src0_1->type) : PXA_PXQ_FMT_NONE;
    const int fmt_g = src0_2 ? pxa_pxq_fmt(src0_2->type) : PXA_PXQ_FMT_NONE;
    if (!src0_2 || !pxqi8_pick_gemm(fmt) || !pxqi8_pick_gemm(fmt_g)) return -1;
    pxq6_maybe_upload_tables(ctx.device);
    pxq23_maybe_upload_books(ctx.device);
    const ggml_unary_op uop = (ggml_unary_op)dst->op_params[0];
    if (uop != GGML_UNARY_OP_SWIGLU_OAI && uop != GGML_UNARY_OP_SILU) return -1;
    if (src1->type != GGML_TYPE_F32 || src1->ne[1] != 1 || src1->ne[3] != 1) return -1;
    if (!ggml_is_contiguous(dst)) return -1;
    const int64_t R = src0_1->ne[1], K = src0_1->ne[0];
    if (R % PXQ4_BM || K % PXQ4_QK || src0_2->ne[0] != K || src0_2->ne[1] != R) return -1;
    if ((size_t)K*sizeof(float) > 46*1024 || (size_t)R*sizeof(float) > 46*1024) return -1;   // stage smem
    if (!pxa_pxq4_bufs_on_device(ctx, {src0_1, src0_2, src1, dst})) return -1;

    cudaStream_t stream = ctx.stream();
    const int64_t n_as  = src0_1->ne[2];
    const int64_t n_ids = ids->ne[0];
    const int   unary = (uop == GGML_UNARY_OP_SWIGLU_OAI) ? 1 : 0;
    const float glu_limit = (uop == GGML_UNARY_OP_SWIGLU_OAI) ? 7.0f : *(const float *)(dst->op_params + 1);

    const int fmt_d = (next && next->op == GGML_OP_MUL_MAT_ID) ? pxa_pxq_fmt(next->src[0]->type) : PXA_PXQ_FMT_NONE;
    const bool fuse_down = next && next->op == GGML_OP_MUL_MAT_ID &&
        pxqi8_pick_gemm(fmt_d) != nullptr && next->src[1] == dst && next->src[2] == ids &&
        next->src[0]->ne[0] == dst->ne[0] && next->src[0]->ne[0] % PXQ4_QK == 0 &&
        next->src[0]->ne[1] % PXQ4_BM == 0 && ggml_is_contiguous(next) &&
        pxa_pxq4_bufs_on_device(ctx, {next->src[0], next});

    ggml_cuda_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool());
    std::vector<int> moe_counts, cum_moe_counts;
    bool is_ser = prepare_row_mappigs(ctx, n_as, n_ids, ids, moe_counts, cum_moe_counts, dev_row_mapping);
    if (is_ser) {
        ggml_tensor * t = fuse_down ? next : dst;
        CUDA_CHECK(cudaMemsetAsync(t->data, 0, ggml_nbytes(t), stream));
    }
    const int64_t total = cum_moe_counts[n_as];
    if (total == 0) return fuse_down ? i + 1 : i;

    std::vector<pxq4_tile_info> tiles;
    tiles.reserve((size_t)(total/PXQ4_BN + n_as + 1));
    for (int e = 0; e < (int)n_as; ++e) {
        for (int t0 = 0; t0 < moe_counts[e]; t0 += PXQ4_BN) {
            tiles.push_back({e, cum_moe_counts[e] + t0, std::min((int)PXQ4_BN, moe_counts[e] - t0), 0});
        }
    }
    if (tiles.empty() || tiles.size() > 65535) return -1;   // grid.y limit; fallback handles it
    if (getenv("PXA_PXQI8_DEBUG")) {
        static long long ncalls = 0;
        if (ncalls++ < 8) {
            int full = 0, ragged_rows = 0;
            for (auto & t : tiles) { if (t.nrows == PXQ4_BN) full++; else ragged_rows += (PXQ4_BN - t.nrows); }
            fprintf(stderr, "PXA_PXQI8_DEBUG: n_as=%lld total_tok=%lld tiles=%zu panels(R/64)=%lld "
                    "full_tiles=%d ragged_tiles=%zu wasted_rows=%d (112-block ceiling on 28-SM 1080Ti)\n",
                    (long long)n_as, (long long)total, tiles.size(), (long long)(R/PXQ4_BM),
                    full, tiles.size()-full, ragged_rows);
        }
    }
    ggml_cuda_pool_alloc<pxq4_tile_info> dev_tiles(ctx.pool(), tiles.size());
    CUDA_CHECK(cudaMemcpyAsync(dev_tiles.get(), tiles.data(), tiles.size()*sizeof(pxq4_tile_info),
                               cudaMemcpyHostToDevice, stream));

    // gather + q8 quantize activations once (per-32 absmax/127 — the stock-MMQ activation class)
    const int kgroups = (int)(K/PXQ4_QK);
    ggml_cuda_pool_alloc<uint8_t> A_q8(ctx.pool(), (size_t)total*K);
    ggml_cuda_pool_alloc<float>   A_d (ctx.pool(), (size_t)total*kgroups);
    k_pxqi8_gather_quant<<<(unsigned)total, 256, (size_t)K*sizeof(float), stream>>>(
            (const char *)src1->data, A_q8.get(), A_d.get(),
            (const pxq4_rowmap *)dev_row_mapping.get(), K, src1->ne[1], src1->nb[1], src1->nb[2]);
    CUDA_CHECK(cudaGetLastError());

    const ggml_tensor * bu = dst->src[4], * bg = dst->src[5];
    dim3 grid((unsigned)(R/PXQ4_BM), (unsigned)tiles.size());
    // N16: the picker also carries the block shape -- the mmq.cuh-geometry twin runs
    // WARP_SIZE x nwarps, the N13/N15 tile runs a flat 64.
    const pxqi8_kern kup = pxqi8_pick_kern(fmt), kgate = pxqi8_pick_kern(fmt_g);
    pxqi8_gemm_fn kern_up = kup.fn, kern_gate = kgate.fn;

    ggml_cuda_pool_alloc<float> C_up(ctx.pool(), (size_t)total*R);
    ggml_cuda_pool_alloc<float> C_gate(ctx.pool(), (size_t)total*R);
    kern_up<<<grid, kup.block, 0, stream>>>((const uint8_t *)src0_1->data, A_q8.get(), A_d.get(), C_up.get(),
            bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0, dev_tiles.get(), (int)R, (int)K);
    kern_gate<<<grid, kgate.block, 0, stream>>>((const uint8_t *)src0_2->data, A_q8.get(), A_d.get(), C_gate.get(),
            bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0, dev_tiles.get(), (int)R, (int)K);
    CUDA_CHECK(cudaGetLastError());

    if (fuse_down) {
        const int64_t Rd = next->src[0]->ne[1];   // Kd == R
        ggml_cuda_pool_alloc<uint8_t> H_q8(ctx.pool(), (size_t)total*R);
        ggml_cuda_pool_alloc<float>   H_d (ctx.pool(), (size_t)total*(R/PXQ4_QK));
        k_pxqi8_glu_quant<<<(unsigned)total, 256, (size_t)R*sizeof(float), stream>>>(
                C_gate.get(), C_up.get(), H_q8.get(), H_d.get(), (int)R, unary, 1.702f, glu_limit);
        CUDA_CHECK(cudaGetLastError());
        ggml_cuda_pool_alloc<float> C_down(ctx.pool(), (size_t)total*Rd);
        dim3 gridd((unsigned)(Rd/PXQ4_BM), (unsigned)tiles.size());
        const pxqi8_kern kdn = pxqi8_pick_kern(fmt_d);
        pxqi8_gemm_fn kern_down = kdn.fn;
        kern_down<<<gridd, kdn.block, 0, stream>>>((const uint8_t *)next->src[0]->data, H_q8.get(), H_d.get(),
                C_down.get(), nullptr, 0, dev_tiles.get(), (int)Rd, (int)R);
        CUDA_CHECK(cudaGetLastError());
        dim3 bd(std::min((unsigned)Rd, 768u));
        k_copy_dst_from_contiguous<<<(unsigned)total, bd, 0, stream>>>((char *)next->data,
                (const char *)C_down.get(), dev_row_mapping.get(), Rd, next->nb[1], next->nb[2]);
        CUDA_CHECK(cudaGetLastError());
        return i + 1;
    }

    // no down fusion: GLU to f32, scatter to dst (existing kernels)
    ggml_cuda_pool_alloc<float> Out(ctx.pool(), (size_t)total*R);
    const int64_t kel = total*R;
    k_pxq4_glu<float><<<(unsigned)((kel + 255)/256), 256, 0, stream>>>(
            C_gate.get(), C_up.get(), Out.get(), kel, unary, 1.702f, glu_limit);
    dim3 bd(std::min((unsigned)R, 768u));
    k_copy_dst_from_contiguous<<<(unsigned)total, bd, 0, stream>>>((char *)dst->data,
            (const char *)Out.get(), dev_row_mapping.get(), R, dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
    return i;
}

#include "ggml-cuda/pxa/pxa_expert_shard.cuh"

static int ggml_cuda_moe_up_gate_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cgraph * graph, int i) {
    ggml_tensor * next = i + 1 < graph->n_nodes ? graph->nodes[i+1] : nullptr;

    const ggml_tensor * src0_1 = dst->src[0];
    const ggml_tensor * src0_2 = dst->src[1];
    const ggml_tensor * src0 = src0_1;
    const ggml_tensor * src1 = dst->src[2];
    const ggml_tensor * ids  = dst->src[3];

    // PXA-SHARD (M2): if the expert weights live in an expert-shard buffer type
    // (M3 loader places the group's *_exps there when PXA_EXPERT_SHARD is set),
    // run the disjoint-row sharded MoE op (no all-reduce) across the matched
    // device group. Flag-off / non-shard parent => the predicate is false and
    // this whole block is skipped (bit-identical to the stock build).
    // PXA-SHARD detection + ntok>1 reconstruct. Holders at FUNCTION scope so the
    // reconstructed full temps + ->data swaps outlive the fall-through stock MMQ enqueue
    // and auto-restore ->data at any function return (RAII).
    ggml_cuda_pool_alloc<char> pxa_ru, pxa_rg;
    pxa_data_swap pxa_sw_u, pxa_sw_g;
    {
        const ggml_tensor * up_root = src0_1;
        while (up_root && up_root->view_src) up_root = up_root->view_src;
        if (pxa_expert_shard_enabled() && up_root && s_pxa_shard_slices.count(up_root->name)) {
            int r = pxa_moe_shard_up_gate_down(ctx, dst, graph, i);
            if (r >= 0) return r;   // ntok==1 decode -> custom parallel kernel: DONE
            // ntok>1 (prefill/reserve): the split weight is a landmine for the stock MMQ
            // below. Reconstruct the full weight on home + repoint src0_1/src0_2->data at it.
            const int home = ctx.device;
            const ggml_tensor * gate_root = src0_2;
            while (gate_root && gate_root->view_src) gate_root = gate_root->view_src;
            cudaStream_t st = ctx.stream(ctx.device, 0);
            if (pxa_shard_reconstruct(ctx, up_root, pxa_ru, home, st) &&
                gate_root && pxa_shard_reconstruct(ctx, gate_root, pxa_rg, home, st)) {
                pxa_sw_u.set(src0_1, pxa_ru.get());
                pxa_sw_g.set(src0_2, pxa_rg.get());
            }
        }
    }

    // PXA_A1 (2026-07-01): fast-TG per-token iy-loop re-reads each routed expert once PER TOKEN of a
    // K-token MTP verify batch. Env-tunable gate: PXA_MOE_FASTTG_MAX_NY=1 routes Ny>1 verify batches to
    // the expert-grouped batched path below (weights read once per traversal). Default 8 = unchanged.
    static const int pxa_fast_tg_max_ny = getenv("PXA_MOE_FASTTG_MAX_NY") ? atoi(getenv("PXA_MOE_FASTTG_MAX_NY")) : 8;
    static const bool pxa_moe_dbg = getenv("PXA_MOE_DEBUG") != nullptr;

    // PXQ4 decode: fused up+gate+swiglu (+down) mmv, f32-in / f32-out, ids on device.
    // Declines (-1) fall through; the q8_1 fast-TG below is gated off for PXQ4 (no vec_dot).
    if (pxa_pxq_fmt(src0_1->type) != PXA_PXQ_FMT_NONE &&
        src1->ne[1] == 1 && src1->ne[2] <= pxa_fast_tg_max_ny && src1->ne[3] == 1) {
        int r = pxa_pxq4_moe_fast_tg(ctx, dst, graph, i);
        if (r >= 0) return r;
    }
    if (pxa_moe_dbg && src1->ne[2] > 1 && src1->ne[2] <= 8) { static int pxa_cnt = 0; if (pxa_cnt < 48) { ++pxa_cnt;
        fprintf(stderr, "PXA_MOE ny=%d fast=%d dev=%d\n", (int)src1->ne[2], src1->ne[2] <= pxa_fast_tg_max_ny ? 1 : 0, ctx.device); } }
    if (src1->ne[1] == 1 && src1->ne[2] <= pxa_fast_tg_max_ny && src1->ne[3] == 1 &&
        ggml_is_quantized(src0_1->type) &&
        src0_1->type != GGML_TYPE_PXQ4 && src0_1->type != GGML_TYPE_PXQ4HQ &&
        src0_1->type != GGML_TYPE_PXQ2 && src0_1->type != GGML_TYPE_PXQ3 &&      // ADD
        src0_1->type != GGML_TYPE_PXQ1 &&
        src0_1->type != GGML_TYPE_PXQ6 &&                                       // ADD (no vec_dot)
        (!src0_2 || ggml_is_quantized(src0_2->type)) &&
        ggml_backend_buffer_is_cuda(src0_1->buffer) &&
        (!src0_2 || ggml_backend_buffer_is_cuda(src0_2->buffer)) &&
        ggml_backend_buffer_is_cuda(src1->buffer) &&
        ggml_backend_buffer_is_cuda(dst->buffer) &&
        src1->type == GGML_TYPE_F32) {
        int device_id = ctx.device;
        ggml_backend_cuda_buffer_context * src0_1_ctx = (ggml_backend_cuda_buffer_context *) src0_1->buffer->context;
        ggml_backend_cuda_buffer_context * src0_2_ctx = src0_2 ? (ggml_backend_cuda_buffer_context *) src0_2->buffer->context : nullptr;
        ggml_backend_cuda_buffer_context * src1_ctx   = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx    = (ggml_backend_cuda_buffer_context *) dst->buffer->context;
        if (src0_1_ctx->device == device_id &&
            (!src0_2_ctx || src0_2_ctx->device == device_id) &&
            src1_ctx->device   == device_id &&
            dst_ctx->device    == device_id) {
            // Fast TG path
            const int64_t n_ids = ids->ne[0];
            auto stream = ctx.stream(device_id, 0);

            auto local_dst = *dst;
            local_dst.ne[2] = n_ids;
            local_dst.ne[1] = local_dst.ne[3] = 1;
            local_dst.nb[1] = local_dst.nb[2] = local_dst.nb[3] = local_dst.ne[0]*sizeof(float);

            auto local_src1 = *src1;
            local_src1.nb[2] = local_src1.nb[3] = 0;
            local_src1.ne[1] = local_src1.ne[2] = local_src1.ne[3] = 1;

            int Ny = src1->ne[2];

            // PXA_A1 E2 (2026-07-06): read-only expert-union saturation probe. On a Ny>1 MTP verify
            // batch, copy routed ids to host and count DISTINCT experts over running prefixes k=1..Ny vs
            // k*n_ids total selections. Decides whether a batched (read-experts-once) verify can pay.
            if (pxa_moe_dbg && Ny > 1 && Ny <= 8 && n_ids > 0 && n_ids <= 32) {
                static int    pxa_e2_calls = 0;
                static double pxa_e2_union[9] = {0};
                static long   pxa_e2_cnt[9]   = {0};
                const int nids = (int)n_ids;
                int32_t hids[32*8];
                for (int t = 0; t < Ny; ++t) {
                    cudaMemcpyAsync(hids + (size_t)t*nids,
                                    (const char *)ids->data + (size_t)t*ids->nb[1],
                                    nids*sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
                }
                cudaStreamSynchronize(stream);
                bool seen[1024]; for (int z = 0; z < 1024; ++z) seen[z] = false;
                int uni = 0;
                for (int k = 1; k <= Ny; ++k) {
                    for (int j = 0; j < nids; ++j) {
                        int e = hids[(size_t)(k-1)*nids + j];
                        if (e >= 0 && e < 1024 && !seen[e]) { seen[e] = true; ++uni; }
                    }
                    pxa_e2_union[k] += uni; pxa_e2_cnt[k] += 1;
                }
                if (++pxa_e2_calls % 200 == 0) {
                    fprintf(stderr, "PXA_E2 calls=%d nids=%d dev=%d union/k:", pxa_e2_calls, nids, ctx.device);
                    for (int k = 1; k <= 8; ++k) if (pxa_e2_cnt[k]) fprintf(stderr, " k%d=%.2f/%d", k, pxa_e2_union[k]/pxa_e2_cnt[k], k*nids);
                    fprintf(stderr, "\n");
                }
            }

            const int64_t src1_padded_col_size = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
            ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool());
            GGML_ASSERT(src1->ne[0] % QK8_1 == 0);
            auto src_1_ddq_size = src1_padded_col_size*sizeof(block_q8_1)/QK8_1;
            local_src1.data = src1_quantized.alloc(src_1_ddq_size * Ny);
            // Note: no use is currently made of the quantization type passed into quantize_row_q8_1_cuda.
            //       If that were to change, we would need to adjust the code to handle src0_1->type != src0_2->type
            quantize_row_q8_1_cuda((const float *)src1->data, (void *)src1_quantized.get(), src1->ne[0], Ny, 1, src1_padded_col_size,
                    src0_1->type, stream);
            CUDA_CHECK(cudaGetLastError());

            local_src1.nb[1] = src_1_ddq_size;

            bool fuse_next = next && next->op == GGML_OP_MUL_MAT_ID && ggml_is_quantized(next->src[0]->type) &&
                ggml_backend_buffer_is_cuda(next->src[0]->buffer) &&
                ((ggml_backend_cuda_buffer_context *)next->src[0]->buffer->context)->device == device_id &&
                ggml_backend_buffer_is_cuda(next->buffer) &&
                ((ggml_backend_cuda_buffer_context *)next->buffer->context)->device == device_id;

            auto unary_op = (ggml_unary_op)dst->op_params[0];
            float limit = *(const float *)(dst->op_params + 1);

            auto local_ids = *ids;
            local_ids.ne[1] = 1;

            // PXA A1 grouped up+gate (env PXA_MOE_BATCHED_VERIFY / legacy PXA_MOE_GROUPED,
            // bit0): read each routed expert ONCE across the Ny verify tokens. v2
            // (2026-07-07): half2 kernel, no host sync, per-device shadow-verify mode,
            // and the v1 gate/up operand-order FIX (src0_1=UP, src0_2=GATE — see
            // grouped_moe_verify.cuh header). Default OFF; one-line env rollback.
            bool pxa_grouped_done = false;
            if (src0_2 && Ny >= 2 && pxa_moe_grouped_flags() != 0 &&
                src0_1->type == GGML_TYPE_MXFP4 && src0_2->type == GGML_TYPE_MXFP4 &&
                unary_op == GGML_UNARY_OP_SILU) {
                pxa_grouped_done = pxa_moe_grouped_gateup(ctx, src0_1, src0_2, src1, ids, dst,
                                                          Ny, (int)n_ids, limit,
                                                          (const char *)src1_quantized.get(), src_1_ddq_size, stream);
            }

            pxa_shard_time_begin(device_id, stream);  // meter UNSHARDED up+gate
            for (int iy = 0; iy < Ny && !pxa_grouped_done; ++iy) {
                local_src1.data = src1_quantized.get() + iy*src_1_ddq_size;
                local_ids.data  = (char *)ids->data + iy*ids->nb[1];
                local_dst.data  = (char *)dst->data + iy*dst->nb[2];
                if (src0_2) {
                    ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, src0_1, &local_src1, &local_ids, &local_dst,
                            dst->src[4], dst->src[5],
                            (const char *)src0_1->data, (const char *)src0_2->data,
                            (const float *)src1->data, (const char *)local_src1.data,
                            (float *)local_dst.data, 0, src0_1->ne[1], 1, src1_padded_col_size, unary_op, limit, stream);
                } else {
                    auto local_src0_1 = *src0_1;
                    local_src0_1.ne[1] /= 2;
                    auto local_src0_2 = local_src0_1;
                    local_src0_2.data = (char *)local_src0_1.data + local_src0_1.ne[1]*local_src0_1.nb[1];
                    if (!dst->src[4]) {
                        ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, &local_src0_1, &local_src1, &local_ids, &local_dst,
                                nullptr, nullptr,
                                (const char *)local_src0_2.data, (const char *)local_src0_1.data,
                                (const float *)src1->data, (const char *)local_src1.data,
                                (float *)local_dst.data, 0, local_src0_1.ne[1], 1, src1_padded_col_size, unary_op, limit, stream);
                    } else {
                        GGML_ASSERT(!dst->src[5]);
                        auto local_bias_1 = *dst->src[4];
                        local_bias_1.ne[0] /= 2;
                        auto local_bias_2 = local_bias_1;
                        local_bias_2.data = (char *)local_bias_1.data + local_bias_1.ne[0]*local_bias_1.nb[0];
                        ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, &local_src0_1, &local_src1, &local_ids, &local_dst,
                                &local_bias_2, &local_bias_1,
                                (const char *)local_src0_2.data, (const char *)local_src0_1.data,
                                (const float *)src1->data, (const char *)local_src1.data,
                                (float *)local_dst.data, 0, local_src0_1.ne[1], 1, src1_padded_col_size, unary_op, limit, stream);
                    }
                }
                CUDA_CHECK(cudaGetLastError());
            }

            pxa_shard_time_end(device_id, stream, "UNSHARDED");  // meter UNSHARDED up+gate
            pxa_moe_grouped_shadow_check(ctx, dst, stream);  // no-op unless PXA_MOE_GROUPED_VERIFY

            if (!fuse_next) return i;

            const int64_t dst_padded_col_size = GGML_PAD(dst->ne[0], MATRIX_ROW_PADDING);
            GGML_ASSERT(dst->ne[0] % QK8_1 == 0);
            auto dst_row_size = dst_padded_col_size*sizeof(block_q8_1)/QK8_1;
            auto dst_ddq_size = n_ids*dst_row_size;
            ggml_cuda_pool_alloc<char> dst_quantized(ctx.pool(), dst_ddq_size*Ny);
            quantize_row_q8_1_cuda((const float *)dst->data, (void *)dst_quantized.get(), dst->ne[0], n_ids*Ny, 1,
                    dst_padded_col_size, next->src[0]->type, stream);
            CUDA_CHECK(cudaGetLastError());

            local_dst.ne[2] = 1;

            auto local_next = *next;
            local_next.ne[2] = local_next.ne[1];
            local_next.ne[1] = local_next.ne[3] = 1;
            local_next.nb[2] = local_next.nb[1];

            local_src1 = *next->src[1];
            local_src1.ne[1] = local_src1.ne[2] = local_src1.ne[3] = 1;
            local_src1.nb[1] = local_src1.nb[2] = local_src1.nb[3] = dst_row_size;

            auto local_src0 = *next->src[0];
            local_src0.ne[2] = local_src0.ne[3] = 1;

            int result = i + 1;

            //printf("next: %ld x %ld x %ld x %ld,    %zu x %zu x %zu x %zu\n", next->ne[0], next->ne[1], next->ne[2], next->ne[3], next->nb[0], next->nb[1], next->nb[2], next->nb[3]);

            for (int iy = 0; iy < Ny; ++iy) {
                local_ids.data  = (char *)ids->data + iy*ids->nb[1];
                auto this_dst_quantized = dst_quantized.get() + iy*dst_ddq_size;
                if (i+2 < graph->n_nodes &&
                    graph->nodes[i+2]->op == GGML_OP_ADD_ID &&
                    graph->nodes[i+2]->src[0] == next &&
                    graph->nodes[i+2]->src[2] == ids) {
                    ggml_cuda_op_mul_mat_vec_q_id(ctx, &local_src0, &local_src1, &local_ids, &local_next, graph->nodes[i+2]->src[1],
                            (const char *)next->src[0]->data, nullptr, this_dst_quantized, (float *)graph->nodes[i+2]->data + iy*next->ne[0]*n_ids,
                            0, next->src[0]->ne[1], 1, dst_padded_col_size, stream);
                    if (iy == 0) {
                        ++result;
                    }
                } else {
                    ggml_cuda_op_mul_mat_vec_q_id(ctx, &local_src0, &local_src1, &local_ids, &local_next, nullptr,
                            (const char *)next->src[0]->data, nullptr, this_dst_quantized, (float *)next->data + iy*next->ne[0]*n_ids,
                            0, next->src[0]->ne[1], 1, dst_padded_col_size, stream);
                }
                CUDA_CHECK(cudaGetLastError());
            }

            return result;
        }
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    ggml_tensor dst_row = *dst;
    // PXA_P100_FP16_GEMM_v1: on a MOE_FUSED_UP_GATE node op_params[0] is the UNARY OP id (silu/
    // swiglu_oai), NOT a ggml_prec — but the per-expert ggml_cuda_mul_mat fallback below hands this
    // local copy to ggml_cuda_op_mul_mat_cublas, which reads op_params[0] as precision and spuriously
    // vetoed the fp16 GemmEx path for every up/gate GEMM (down-proj, a real MUL_MAT_ID node with
    // op_params[0]=0, got fp16 — measured 2/3 of expert GEMMs stuck on fp32 SGEMM). The unary op is
    // read from the ORIGINAL dst above; clearing it on the local copy is safe for every consumer here.
    dst_row.op_params[0] = GGML_PREC_DEFAULT;

    // PXQ4 prefill: ONE grouped fused GEMM launch per projection over ALL routed experts
    // (replaces the per-expert dequant+cublas loop below — the PoC-measured 2.53x P100 path).
    if (pxa_pxq_fmt(src0_1->type) != PXA_PXQ_FMT_NONE) {
        int r = pxa_pxq_moe_prefill_i8(ctx, dst, graph, i);   // N13 (env-gated, default OFF)
        if (r < 0) r = pxa_pxq4_moe_prefill(ctx, dst, graph, i);
        if (r >= 0) return r;
    }

    // The heuristics src1->ne[2] <= 32*src0->ne[2] to use the mul_mat_id implementation instead of the original version
    // is derived from
    //    * DeepSeek-Lite:  64 total, 6 active experts
    //    * GPT-OSS-20B  :  32 total, 4 active experts
    //    * Qwen3-30B-A3B: 128 total, 8 active experts
    // My original hypothesis was that it is dependent on the total/active experts ratio, but from these 3 it
    // looks like it really depends just on the total number of experts.
    // TODO: verify with more models, or perhaps make the magic constant '32' to be defined via a compile time define.
    if (src1->ne[2] <= ctx.mmq_id_thresh*src0->ne[2] &&
        ggml_is_quantized(src0_1->type) && (!src0_2 || src0_1->type == src0_2->type) && src1->ne[1] == 1 && src1->ne[3] == 1 &&
        ggml_cuda_can_use_mmq_id(src0_1->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[2])) {

        const int64_t ne_get_rows = ne12 * n_ids;
        ggml_cuda_pool_alloc<int32_t> ids_device(ctx.pool(), ne_get_rows + ne_get_rows + n_as + 1);
        auto ids_src1 = ids_device.get();
        auto ids_dst  = ids_src1 + ne_get_rows;
        auto expert_bounds = ids_dst + ne_get_rows;

        compute_row_ids((const int32_t *)ids->data, ids_src1, ids_dst, expert_bounds,
                ne02, ne12, n_ids, ne11, nb11, nb12, ids->nb[1], stream);

        const int64_t ne11_flat = ne12*n_ids;
        const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
        size_t nbytes_src1_q8_1 = ne11_flat*ne10_padded * sizeof(block_q8_1)/QK8_1 +
                get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), nbytes_src1_q8_1);

        size_t ts_src1 = ggml_type_size(src1->type);
        quantize_mmq_q8_1_cuda_id((const float *)src1->data, ids_src1, src1_quantized.get(),
                src0_1->type, ne10, src1->nb[1] / ts_src1, src1->nb[2] / ts_src1, src1->nb[2] / ts_src1,
                ne10_padded, ne11_flat, 1, 1, stream);

        if (src0_2) {
        ggml_cuda_pool_alloc<char> dst_up_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));
        ggml_cuda_pool_alloc<char> dst_gate_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));

        dst_row.data = dst_up_contiguous.get();
        ggml_cuda_mul_mat_q_id(ctx, src0_1, src1, ids, &dst_row, (char *)ids_device.get(), src1_quantized.get());
        if (dst->src[4]) {
            ggml_cuda_add_id((const float *)dst_row.data, (const float *)dst->src[4]->data, (const int32_t *)ids->data,
                    (float *)dst_row.data, dst_row.ne[0], dst_row.ne[1], dst_row.ne[2], dst_row.ne[0], dst_row.ne[1],
                    dst_row.nb[1], dst_row.nb[2], dst->src[4]->nb[1], ids->nb[1], stream);
            CUDA_CHECK(cudaGetLastError());
        }

        dst_row.data = dst_gate_contiguous.get();
        ggml_cuda_mul_mat_q_id(ctx, src0_2, src1, ids, &dst_row, (char *)ids_device.get(), src1_quantized.get());
        if (dst->src[5]) {
            ggml_cuda_add_id((const float *)dst_row.data, (const float *)dst->src[5]->data, (const int32_t *)ids->data,
                    (float *)dst_row.data, dst_row.ne[0], dst_row.ne[1], dst_row.ne[2], dst_row.ne[0], dst_row.ne[1],
                    dst_row.nb[1], dst_row.nb[2], dst->src[5]->nb[1], ids->nb[1], stream);
            CUDA_CHECK(cudaGetLastError());
        }

        auto unary_op = (ggml_unary_op)dst->op_params[0];
        if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) {
            ggml_swiglu_oai_cuda_f32((const float *)dst_gate_contiguous.get(), (const float *)dst_up_contiguous.get(),
                        (float *)dst->data, ggml_nelements(dst), dst_row.ne[0],  dst_row.ne[0],  dst_row.ne[0],
                        1.702f, 7.0f, stream);
        } else {
            float limit = *((const float *)(dst->op_params + 1));
            //printf("%s: using limit = %g\n", __func__, limit);
            ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(&dst_row),
                    (const float *)dst_gate_contiguous.get(), (const float *)dst_up_contiguous.get(),
                    (float *)dst->data, limit);
        }
        } else {

            ggml_cuda_pool_alloc<char> dst_up_gate_contiguous(ctx.pool(), 2*sizeof(float)*ggml_nelements(dst));
            ggml_cuda_pool_alloc<char> dst_gate_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));
            dst_row.ne[0] *= 2;
            dst_row.nb[1] *= 2;
            dst_row.nb[2] *= 2;
            dst_row.nb[3] *= 2;
            dst_row.data = dst_up_gate_contiguous.get();
            ggml_cuda_mul_mat_q_id(ctx, src0_1, src1, ids, &dst_row, (char *)ids_device.get(), src1_quantized.get());
            if (dst->src[4]) {
                GGML_ASSERT(!dst->src[5]);
                ggml_cuda_add_id((const float *)dst_row.data, (const float *)dst->src[4]->data, (const int32_t *)ids->data,
                        (float *)dst_row.data, dst_row.ne[0], dst_row.ne[1], dst_row.ne[2], dst_row.ne[0], dst_row.ne[1],
                        dst_row.nb[1], dst_row.nb[2], dst->src[4]->nb[1], ids->nb[1], stream);
                CUDA_CHECK(cudaGetLastError());
            }

            auto unary_op = (ggml_unary_op)dst->op_params[0];
            if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) {
                ggml_swiglu_oai_cuda_f32((const float *)dst_up_gate_contiguous.get(), (const float *)dst_up_gate_contiguous.get() + dst->ne[0],
                        (float *)dst->data, ggml_nelements(dst), dst->ne[0], src0_1->ne[1], src0_1->ne[1],
                        1.702f, 7.0f, stream);
            } else {
                float limit = *((const float *)(dst->op_params + 1));
                //printf("%s: using limit = %g\n", __func__, limit);
                ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(dst), dst->ne[0],
                        (const float *)dst_up_gate_contiguous.get(), (float *)dst->data, limit);
            }
        }
        CUDA_CHECK(cudaGetLastError());

        if (next && next->op == GGML_OP_MUL_MAT_ID && ggml_is_quantized(next->src[0]->type) &&
            ggml_cuda_should_use_mmq(next->src[0]->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[2])) {
            //ggml_cuda_mul_mat_q_id(ctx, next->src[0], dst, ids, next, (char *)ids_device.get(), nullptr);
            ggml_cuda_mul_mat_q_id(ctx, next->src[0], dst, ids, next, nullptr, nullptr);
            return i+1;
        }

        return i;
    }

    ggml_tensor src0_1_row = *src0_1;
    ggml_tensor src0_2_row; if (src0_2) src0_2_row = *src0_2;
    ggml_tensor src1_row   = *src1;
    ggml_tensor final_dst;
    ggml_tensor final_src;

    char * src0_1_original = (char *) src0_1->data;
    char * src0_2_original = src0_2 ? (char *) src0_2->data : nullptr;
    char * src1_original   = (char *) src1->data;
    char * dst_original    = (char *)  dst->data;

    src0_1_row.ne[2] = 1;
    src0_1_row.ne[3] = 1;
    src0_1_row.nb[3] = nb02;
    if (src0_2) {
        src0_2_row.ne[2] = 1;
        src0_2_row.ne[3] = 1;
        src0_2_row.nb[3] = nb02;
    }

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;

    bool fuse_down = false;
    if (next && next->op == GGML_OP_MUL_MAT_ID) {
        fuse_down = true;
        final_dst = *next;
        final_dst.ne[1] = final_dst.ne[2] = final_dst.ne[3] = 1;
        final_dst.nb[2] = final_dst.nb[3] = final_dst.nb[1];
        final_src = *next->src[0];
        final_src.ne[2] = final_src.ne[3] = 1;
        final_src.nb[3] = final_src.nb[2];
    }

    ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool());
    bool use_quantized_src1 = false;
    int64_t src1_padded_num_cols = 0, src1_padded_row_size = 0, src1_quantized_size = 0;
    if (ggml_is_quantized(src0_1->type) && (!src0_2 || src0_1->type == src0_2->type) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        if (ggml_cuda_should_use_mmq(src0_1->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[2])) {
            src1_padded_num_cols = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
            src1_padded_row_size = src1_padded_num_cols/ggml_blck_size(GGML_TYPE_Q8_1)*ggml_type_size(GGML_TYPE_Q8_1);
            src1_quantized_size  = src1_padded_row_size*src1->ne[2] + get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
            src1_quantized.alloc(src1_quantized_size);
            use_quantized_src1 = true;
        }
    }
    ggml_cuda_pool_alloc<char> src1_contiguous(ctx.pool());
    if (!use_quantized_src1) {
        src1_contiguous.alloc(sizeof(float)*ggml_nelements(src1));
    }
    ggml_cuda_pool_alloc<char> dst_up_contiguous(ctx.pool()), dst_gate_contiguous(ctx.pool());
    if (src0_2) {
        dst_up_contiguous.alloc(sizeof(float)*ggml_nelements(dst));
        dst_gate_contiguous.alloc(sizeof(float)*ggml_nelements(dst));
    } else {
        dst_up_contiguous.alloc(2*sizeof(float)*ggml_nelements(dst));
        dst_gate_contiguous.alloc(sizeof(float)*ggml_nelements(dst));
    }
    ggml_cuda_pool_alloc<char> final_dst_contiguous(ctx.pool());
    if (fuse_down) {
        final_dst.data = final_dst_contiguous.alloc(sizeof(float)*ggml_nelements(next));
        final_dst.src[1] = &dst_row;
    }

    src1_row.data = src1_contiguous.get();

    ggml_cuda_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool());
    std::vector<int> moe_counts, cum_moe_counts;

    bool is_ser = prepare_row_mappigs(ctx, n_as, n_ids, ids, moe_counts, cum_moe_counts, dev_row_mapping);
    if (is_ser) {
        if (fuse_down) {
            CUDA_CHECK(cudaMemsetAsync(next->data, 0, ggml_nbytes(next), stream));
        } else {
            CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
        }
    }

    for (int64_t i02 = 0; i02 < n_as; i02++) {
        int64_t num_src1_rows = moe_counts[i02];

        if (num_src1_rows == 0) continue;
        size_t mapping_offset = cum_moe_counts[i02];

        if (use_quantized_src1) {
            quantize_mmq_q8_1_id_cuda((const float *)src1->data, src1_quantized.get(), (const char *)(dev_row_mapping.get() + mapping_offset),
                    src1->ne[0], num_src1_rows, src1_padded_num_cols, src0_1->type, stream);
            CUDA_CHECK(cudaGetLastError());
            src1_row.data = src1_quantized.get();
        }
        else {
            dim3 block_dims(std::min((unsigned int)ne10, 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_src_to_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    src1_original, src1_contiguous.get(), dev_row_mapping.get() + mapping_offset, ne10, ne11, nb11, nb12);
            CUDA_CHECK(cudaGetLastError());
            src1_row.data = src1_contiguous.get();
        }

        src0_1_row.data = src0_1_original + i02*nb02;
        if (src0_2_original) src0_2_row.data = src0_2_original + i02*src0_2->nb[2];   // gate stride, NOT up nb02 (mixed PXQ pairs differ)

        GGML_ASSERT(nb11 == sizeof(float)*ne10);
        GGML_ASSERT(nb1 == sizeof(float)*ne0);

        auto nb1l = nb1;
        if (!src0_2) {
            nb1l = nb1*2;
            dst_row.ne[0] = dst->ne[0] * 2;
        }

        src1_row.ne[1] = num_src1_rows;
        src1_row.nb[1] = use_quantized_src1 ? src1_padded_row_size : nb11;
        src1_row.nb[2] = num_src1_rows*src1_row.nb[1];
        src1_row.nb[3] = num_src1_rows*src1_row.nb[1];

        dst_row.ne[1] = num_src1_rows;
        dst_row.nb[1] = nb1l;
        dst_row.nb[2] = num_src1_rows*nb1l;
        dst_row.nb[3] = num_src1_rows*nb1l;

        dst_row.data  =  dst_up_contiguous.get();
        if (use_quantized_src1) {
            ggml_cuda_mul_mat_q_id(ctx, &src0_1_row, &src1_row, nullptr, &dst_row, nullptr, src1_quantized.get());
        } else {
            ggml_cuda_mul_mat(ctx, &src0_1_row, &src1_row, &dst_row, nullptr, 0);
        }
        CUDA_CHECK(cudaGetLastError());

        if (dst->src[4]) {
            GGML_ASSERT(dst_row.ne[0] == dst->src[4]->ne[0]);
            dim3 block_dims(std::min(uint32_t(dst_row.ne[0]), 768u));
            dim3 grid_dims(num_src1_rows);
            k_quick_add<<<grid_dims, block_dims, 0, stream>>>(dst_row.ne[0], (const float *)dst_row.data,
                    (const float *)((const char *)dst->src[4]->data + i02*dst->src[4]->nb[1]), (float *)dst_row.data);
            CUDA_CHECK(cudaGetLastError());
        }

        auto unary_op = (ggml_unary_op)dst->op_params[0];
        float limit = *(const float *)(dst->op_params + 1);
        //printf("%s: using limit = %g\n", __func__, limit);
        if (src0_2) {
            dst_row.data  = dst_gate_contiguous.get();
            if (use_quantized_src1) {
                ggml_cuda_mul_mat_q_id(ctx, &src0_2_row, &src1_row, nullptr, &dst_row, nullptr, src1_quantized.get());
            } else {
                ggml_cuda_mul_mat(ctx, &src0_2_row, &src1_row, &dst_row, nullptr, 0);
            }
            CUDA_CHECK(cudaGetLastError());

            if (dst->src[5]) {
                dim3 block_dims(std::min(uint32_t(dst_row.ne[0]), 768u));
                dim3 grid_dims(num_src1_rows);
                k_quick_add<<<grid_dims, block_dims, 0, stream>>>(dst_row.ne[0], (const float *)dst_row.data,
                        (const float *)((const char *)dst->src[5]->data + i02*dst->src[5]->nb[1]), (float *)dst_row.data);
                CUDA_CHECK(cudaGetLastError());
            }
            if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) {
                ggml_swiglu_oai_cuda_f32((const float *)dst_gate_contiguous.get(), (const float *)dst_up_contiguous.get(),
                        (float *)dst_gate_contiguous.get(), ggml_nelements(&dst_row), dst_row.ne[0],  dst_row.ne[0],  dst_row.ne[0],
                        1.702f, 7.0f, stream);
            } else {
                ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(&dst_row),
                        (const float *)dst_gate_contiguous.get(), (const float *)dst_up_contiguous.get(),
                        (float *)dst_gate_contiguous.get(), limit);
            }
        } else {
            if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) {
                ggml_swiglu_oai_cuda_f32((const float *)dst_up_contiguous.get(), (const float *)dst_up_contiguous.get() + dst->ne[0],
                        (float *)dst_gate_contiguous.get(), ggml_nelements(&dst_row)/2, dst->ne[0], src0_1->ne[1], src0_1->ne[1],
                        1.702f, 7.0f, stream);
            } else {
                ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(&dst_row)/2, dst->ne[0],
                        (const float *)dst_up_contiguous.get(), (float *)dst_gate_contiguous.get(), limit);
            }
            dst_row.data = dst_gate_contiguous.get();
            dst_row.ne[0] /= 2;
            dst_row.nb[1] /= 2;
            dst_row.nb[2] /= 2;
            dst_row.nb[3] /= 2;
        }
        CUDA_CHECK(cudaGetLastError());

        if (fuse_down) {

            final_dst.ne[1] = num_src1_rows;
            final_dst.nb[1] = final_dst.ne[0]*sizeof(float);
            final_dst.nb[2] = final_dst.nb[3] = num_src1_rows*final_dst.nb[1];
            final_src.data = (char *)next->src[0]->data + i02*next->src[0]->nb[2];
            if (ggml_is_quantized(next->src[0]->type) &&
                ggml_cuda_should_use_mmq(final_src.type, ggml_cuda_info().devices[ctx.device].cc, dst_row.ne[1])) {
                ggml_cuda_mul_mat_q_id(ctx, &final_src, &dst_row, nullptr, &final_dst, nullptr, nullptr);
            } else {
                ggml_cuda_mul_mat(ctx, &final_src, &dst_row, &final_dst, nullptr, 0);
            }
            CUDA_CHECK(cudaGetLastError());

            dim3 block_dims(std::min((unsigned int)next->ne[0], 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_dst_from_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    (char *)next->data, final_dst_contiguous.get(),
                    dev_row_mapping.get() + mapping_offset,
                    next->ne[0],
                    next->nb[1], next->nb[2]);
            CUDA_CHECK(cudaGetLastError());

        }
        else {

            dim3 block_dims(std::min((unsigned int)ne0, 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_dst_from_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    dst_original, dst_gate_contiguous.get(),
                    dev_row_mapping.get() + mapping_offset,
                    ne0,
                    nb1, nb2);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    return fuse_down ? i+1 : i;
}

static inline bool pxa_is_pxq_type(ggml_type t) {
    return t == GGML_TYPE_PXQ4 ||
           t == GGML_TYPE_PXQ4HQ || t == GGML_TYPE_PXQ2 || t == GGML_TYPE_PXQ3 ||
           t == GGML_TYPE_PXQ1 ||
           t == GGML_TYPE_PXQ6;
}

// ================= DENSE FUSED up+gate+GLU decode driver (PXA_PXQ_DENSE_GATEUP) =================
// A dense GGML_OP_FUSED_UP_GATE node whose up/gate are PXQ-slab-typed had NO fused kernel: the
// handler below split it into two independent ggml_cuda_mul_mat calls (two k_pxq6_mmv launches,
// each staging the WHOLE x vector into smem and each carrying its own tab/anchor prologue) plus a
// third elementwise GLU kernel over the R-wide intermediate. The MoE path has ridden the fused
// k_pxq6_gateup_mmv family since day one; the dense path simply never reached it, even though the
// kernel is expert-agnostic (the one-entry {0} ids idiom already used by pxa_pxq_mmv_2d degenerates
// pxq6_panel() to W + p*panel_stride).
//
// The mechanism that matters at decode is NOT the saved launches: this driver is register-limited
// and latency-bound (ncu: 19.8-38.6% achieved occupancy, Block Limit = Registers, DRAM 6-36% of
// peak), so the win is memory-level parallelism -- one thread issuing the up-slab and gate-slab
// loads back to back doubles the outstanding loads per thread at identical weight bytes.
//
// Both launch geometries are supported so this composes with the K8-2D S-split rather than
// undoing it: with S == 1 it is the plain k_pxq6_gateup_mmv, above 1 the S-way chunk form plus
// k_pxq6_gateup_reduce_gen (the gen workspace reserves 8 chunk slots, so S is capped at 8).
static bool pxa_pxq_dense_gateup() {
    static const bool v = [] {
        const char * e = getenv("PXA_PXQ_DENSE_GATEUP");
        return !(e && atoi(e) == 0);
    }();
    return v;
}

// Same phantom-lever discipline as the K1/K8 logs: a driver that silently declines is
// indistinguishable from one that fires, so record the first of each per device.
static void pxa_pxq_dense_gateup_log(int device, bool fired, int S, int panels, int ny,
                                     int R, int K, const char * why) {
    static std::atomic<uint32_t> seen{0};
    if (device < 0 || device >= 16) return;
    const uint32_t bit = 1u << (2*device + (fired ? 0 : 1));
    if (seen.fetch_or(bit) & bit) return;
    fprintf(stderr, "PXA_PXQ_DENSE_GATEUP dev%d: %s (S=%d panels=%d ny=%d R=%d K=%d)%s%s\n",
            device, fired ? "FIRING" : "DECLINED -> split up/gate mul_mat", S, panels, ny, R, K,
            why ? " why=" : "", why ? why : "");
}

// Independent split control for the FUSED dense gateup. The shared K8-2D target is tuned on the
// single-weight mmv, whose grids are half the size (one weight, not two): at the sm_70 knee the
// fused gateup already launches panels = R/64 = 272 blocks on 80 SMs before any split, so it may
// want a different S -- and every S > 1 buys its occupancy at the price of one dependent
// k_pxq6_gateup_reduce_gen. 1 = follow the shared target (default), 0 = never split the fused
// gateup; PXA_PXQ_DENSE_GATEUP_TARGET overrides the block target for this node class only.
static bool pxa_pxq_dense_gateup_split() {
    static const bool v = [] {
        const char * e = getenv("PXA_PXQ_DENSE_GATEUP_SPLIT");
        return !(e && atoi(e) == 0);
    }();
    return v;
}
// MEASURED 2026-07-27 (Qwable-27B dense, PXQ4core, 4 timed reps/arm, spread < 0.2%): the two
// target Teslas want OPPOSITE answers here, which is why this is its own knob.
//   2xV100 (sm_70): follow the shared target 33.618 t/s | never split 33.905  (+0.85%)
//   2xP100 (sm_60): follow the shared target 20.793 t/s | never split 18.997  (-8.6%)
// The fused gateup carries TWO weights per block, so panels = R/64 blocks already do the work
// of 2*panels mmv blocks. On sm_70 (80 SMs) the dense ffn_up/gate grid is 272 blocks, which is
// already past the point where more parallelism pays, so every S > 1 buys nothing and still
// costs a dependent k_pxq6_gateup_reduce_gen. On sm_60 (56 SMs) the same grid is well short of
// the machine's appetite and the split is strongly positive. Other arches follow the shared
// target (unmeasured -- do not guess). 0 == never split.
static int pxa_pxq_dense_gateup_target(int device) {
    static const int ov = [] {
        const char * e = getenv("PXA_PXQ_DENSE_GATEUP_TARGET");
        return e ? atoi(e) : -1;
    }();
    if (ov >= 0) return ov;
    if (ggml_cuda_info().devices[device].cc == 700) return 0;   // sm_70: never split the fused gateup
    return pxa_pxq4_2d_split_target(device);
}

// returns 0 if it handled the node, -1 to decline (caller falls through to the split path)
static int pxa_pxq_gateup_2d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!pxa_pxq_dense_gateup() || !pxa_pxq4_2d_enabled()) return -1;

    const ggml_tensor * src0_1 = dst->src[0];   // up
    const ggml_tensor * src0_2 = dst->src[1];   // gate
    const ggml_tensor * src1   = dst->src[2];   // x
    if (!src0_1 || !src0_2 || !src1) return -1;

    const int fmt   = pxa_pxq_fmt(src0_1->type);
    const int fmt_g = pxa_pxq_fmt(src0_2->type);
    if (fmt == PXA_PXQ_FMT_NONE || fmt_g == PXA_PXQ_FMT_NONE) return -1;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return -1;

    // pxq4_glu_apply covers SILU-swiglu and SWIGLU_OAI. The split path it replaces routes the
    // op through ggml_fused_mul_unary, which asserts on anything but SILU/RELU/GELU -- so SILU
    // is the only op both forms can serve. Keep the fused path to exactly that.
    const ggml_unary_op uop = (ggml_unary_op)dst->op_params[0];
    if (uop != GGML_UNARY_OP_SILU) return -1;

    // dense: no expert axis anywhere, no ids, no per-expert bias sources
    if (src0_1->ne[2] != 1 || src0_1->ne[3] != 1) return -1;
    if (src0_2->ne[2] != 1 || src0_2->ne[3] != 1) return -1;
    if (src1->ne[2]   != 1 || src1->ne[3]   != 1) return -1;
    if (dst->ne[2]    != 1 || dst->ne[3]    != 1) return -1;
    if (dst->src[3] || dst->src[4] || dst->src[5]) return -1;
    if (!ggml_is_contiguous(src0_1) || !ggml_is_contiguous(src0_2) ||
        !ggml_is_contiguous(src1)   || !ggml_is_contiguous(dst)) return -1;

    const int64_t K = src0_1->ne[0], R = src0_1->ne[1];
    if (src0_2->ne[0] != K || src0_2->ne[1] != R) return -1;
    if (K % PXQ6_QK || R % PXQ6_BM) return -1;
    if (src1->ne[0] != K || dst->ne[0] != R) return -1;

    const int64_t ny = src1->ne[1];
    if (ny < 1 || ny > pxa_pxq4_2d_max_ny()) return -1;
    if (dst->ne[1] != ny) return -1;

    if (!pxa_pxq4_bufs_on_device(ctx, {src0_1, src0_2, src1, dst})) return -1;

    if (fmt >= PXA_PXQ_FMT_P6 || fmt_g >= PXA_PXQ_FMT_P6) pxq6_maybe_upload_tables(ctx.device);
    if (fmt >= PXA_PXQ_FMT_P2 || fmt_g >= PXA_PXQ_FMT_P2) pxq23_maybe_upload_books(ctx.device);

    const int  pair = pxa_pxq6_decode_mode();
    const bool vecx = pxa_pxq6_vecx();

    cudaStream_t stream = ctx.stream();
    const int    kslabs = (int)(K/PXQ6_QK);
    const int    panels = (int)(R/PXQ6_BM);
    const float  limit  = *(const float *)(dst->op_params + 1);

    // one-entry ids buffer == {0}; both strides zero so every (token, slot) block reads it.
    const int32_t * idz = pxa_pxq_zero_ids(ctx.device, stream);
    ggml_cuda_pool_alloc<int32_t> ids_pool(ctx.pool());
    if (!idz) {
        ids_pool.alloc(1);
        CUDA_CHECK(cudaMemsetAsync(ids_pool.get(), 0, sizeof(int32_t), stream));
        idz = ids_pool.get();
    }

    // ---- S-split form: same occupancy target the K8-2D mmv split aims at, capped at the
    // canonical chunk count (PXQ_CANON_v1: S must divide NFIX so the split stays bit-exact).
    const int nfix_gu = pxq6_canon_nfix(kslabs, PXQ6_GU_SPLIT_MAX);
    if (pxa_pxq4_2d_split() && pxa_pxq_dense_gateup_split()) {
        const int target = pxa_pxq_dense_gateup_target(ctx.device);
        int S = 1;
        while (S < nfix_gu && (int64_t)panels*ny*S < target) S *= 2;
        if (S > 1) {
            auto * ksg = pxq6_pick_gateup_ksplit_gen(fmt, fmt_g, pair, vecx);
            int nthr_dgu = 256;
            if (ksg && pxa_pxq6_rowx2()) {
                auto * k2 = pxq6_pick_gateup_ksplit_gen_x2(fmt, fmt_g, pair, vecx);
                pxa_pxq6_rowx2_log(3, k2 != nullptr);
                if (k2) { ksg = k2; nthr_dgu = 128; }
            }
            if (ksg && nthr_dgu == 256 && (pxa_pxq6_qpf() & 2)) {   // Q1 yields to an engaged R2 twin
                auto * kq = pxq6_pick_gateup_ksplit_gen_qp(fmt, fmt_g, pair, vecx);
                pxa_pxq6_qpf_log(3, kq != nullptr);
                if (kq) ksg = kq;
            }
            const int    kc_max  = ((kslabs + S - 1)/S)*PXQ6_QK;
            const size_t smem_s  = (size_t)kc_max*sizeof(float);
            if (ksg && smem_s <= 46*1024) {
                // PXQ_CANON_v1 workspace: raw per-(fixed-chunk, lane) partials, u then g halves
                const size_t need = (size_t)ny*2*(size_t)(PXQ6_GU_SPLIT_MAX*PXQ4_MMV_KSEG)*(size_t)R;
                float * ws = pxq6_ksplit_workspace(ctx.device, stream, need);
                if (ws) {
                    pxa_pxq_dense_gateup_log(ctx.device, true, S, panels, (int)ny, (int)R, (int)K, nullptr);
                    int * ctrs = pxa_pxq_split_fusered()
                        ? pxq6_ksplit_counters(ctx.device, stream) : nullptr;
                    dim3 grids((unsigned)(panels*S), 1u, (unsigned)ny);
                    ksg<<<grids, nthr_dgu, smem_s, stream>>>(
                        (const uint8_t *)src0_1->data, (const uint8_t *)src0_2->data,
                        (const char *)src1->data, src1->nb[1], ws,
                        (const char *)idz, /* ids_nb0 */ 0, /* ids_nb1 */ 0,
                        (char *)dst->data, dst->nb[1], /* dst_slot_stride */ 0,
                        /* bias_u */ nullptr, 0, /* bias_g */ nullptr, 0,
                        ctrs, /* unary SILU */ 0, 1.702f, limit,
                        (int)R, (int)K, /* n_as */ 1, /* n_ids */ 1, S);
                    CUDA_CHECK(cudaGetLastError());
                    if (!ctrs) {
                        const int rblk = pxa_pxq_reduce_blk();
                        dim3 gridr((unsigned)((R + rblk - 1)/rblk), 1u, (unsigned)ny);
                        k_pxq6_gateup_reduce_gen<<<gridr, rblk, 0, stream>>>(ws,
                            (char *)dst->data, dst->nb[1], /* dst_slot_stride */ 0,
                            (const char *)idz, /* ids_nb0 */ 0, /* ids_nb1 */ 0,
                            /* bias_u */ nullptr, 0, /* bias_g */ nullptr, 0,
                            (int)R, /* n_as */ 1, /* n_ids */ 1, /* unary SILU */ 0, 1.702f, limit, kslabs);
                        CUDA_CHECK(cudaGetLastError());
                    }
                    return 0;
                }
            }
        }
    }

    // ---- plain form: stages the whole x vector, so it owns the same K cap as the K1-2D mmv.
    const size_t smem_gu = (size_t)K*sizeof(float) + 2*PXQ4_MMV_KSEG*64*sizeof(float);
    if (smem_gu > 46*1024) {
        pxa_pxq_dense_gateup_log(ctx.device, false, 1, panels, (int)ny, (int)R, (int)K, "x-stage>46K");
        return -1;
    }
    auto * kern = pxq6_pick_gateup(fmt, fmt_g, pair, vecx);
    if (!kern) {
        pxa_pxq_dense_gateup_log(ctx.device, false, 1, panels, (int)ny, (int)R, (int)K, "no-instantiation");
        return -1;
    }
    pxa_pxq_dense_gateup_log(ctx.device, true, 1, panels, (int)ny, (int)R, (int)K, nullptr);
    dim3 grid((unsigned)panels, 1u, (unsigned)ny);
    kern<<<grid, 256, smem_gu, stream>>>(
        (const uint8_t *)src0_1->data, (const uint8_t *)src0_2->data,
        (const char *)src1->data, src1->nb[1],
        (char *)dst->data, dst->nb[1], /* dst_slot_stride */ 0,
        (const char *)idz, /* ids_nb0 */ 0, /* ids_nb1 */ 0,
        /* bias_u */ nullptr, 0, /* bias_g */ nullptr, 0,
        (int)R, (int)K, /* n_as */ 1, /* unary SILU */ 0, 1.702f, limit);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

static void ggml_cuda_up_gate_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0_1 = dst->src[0];
    const ggml_tensor * src0_2 = dst->src[1];
    const ggml_tensor * src1 = dst->src[2];

    // Dense FUSED_UP_GATE with PXQ-slab-typed up/gate tensors. The stock q8_1 mmvq/mmq calls
    // below have NO PXQ kernels and would fault; divert to a per-operand ggml_cuda_mul_mat
    // (which reaches the PXQ 2D drivers, and the dequant->cublas fallback beyond their ny
    // window) plus the fused GLU -- correct at any ne11.
    // NOTE: since BACKBONE_REV 2 this is a NORMAL path, not just a surgery guard: the rev-2
    // backbone table puts ffn_{up,gate}_shexp and dense ffn_{up,gate} on native PXQ types, so
    // every rev-2 file reaches it. (The older "no file from this quantizer gets here" note is
    // obsolete.)
    // PXA_PXQ_MMVQ: when both operands are an MMVQ-registered PXQ tier at decode width, fall
    // through to the generic path below -- it reaches ggml_cuda_op_fused_mul_mat_vec_q_id, i.e.
    // ONE kernel that walks the activation once for up and gate together (the fusion MXFP4 gets
    // and the per-operand divert below throws away).
    const bool pxq_mmvq_fug = pxa_pxq_mmvq_type(src0_1->type) && src0_1->type == src0_2->type
        && pxa_pxq_mmvq_on(ggml_cuda_info().devices[ctx.device].cc)
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE && src1->ne[2] == 1 && src1->ne[3] == 1
        && src0_1->ne[2] == 1 && src0_1->ne[3] == 1 && src1->type == GGML_TYPE_F32;
    // PXA_PXQ_MMVQ_FUGSPLIT (2026-08-10, default OFF): run the dense PXQ up/gate decode as TWO
    // plain MMVQ launches + the fused GLU epilogue instead of the fused_mul_mat_vec_q twin.
    // WHY: on sm_70 the fused twin doubles the per-thread state (two weight streams x ROWS
    // anchors/scales/accumulators) -> 92 regs + 64 B stack spills at the ROWS=4 shape the plain
    // kernel wants -> 31%% occupancy, measured 248 us/call where Q4_K's fused twin (48 regs, one
    // superblock scale) runs 175 us. The plain PXQ4 kernel (63 regs, no spills) is at per-byte
    // parity with Q4_K's plain kernel, so two plain launches beat one crippled fused launch.
    // The two streams share the q8_1 sidecar; the GLU epilogue is the existing fused_mul_unary.
    // Numerics: same vec_dot, same per-stream fold shape, same GLU arithmetic -> expected
    // bit-identical to the fused twin; gated by hash before any default flip.
    // DEFAULT ON for sm_70+ as of 2026-09-02. The fused twin holds TWO PXQ weight streams at the
    // row tile the PLAIN kernel was tuned for, which on Volta costs 92 registers plus 64 B of
    // stack spill and drops the kernel to 31% occupancy; two plain MMVQ launches plus the
    // existing GLU epilogue move the same weight bytes with none of that. The dense ffn up/gate
    // pair is the largest single weight chunk in the graph (2 of every 3 FFN bytes, 65 layers
    // here), so this is where the codec was losing its decode. See LEVERS.md for the measured
    // arms. sm_60 is unaffected: P100 has no DP4A and pxa_pxq_mmvq_on() never admits it here.
    // Numerics: NOT bit-identical -- MEASURED, correcting the 2026-08-10 prediction above. Both
    // paths run the same vec_dot over the same weight bytes, but the split reduces up and gate in
    // two separate launches and feeds the GLU from f32 memory instead of the twin's live
    // registers, and that reassociation is observable. On the 20-prompt temp-0 greedy set
    // (Qwable-27B PXQ4-core, 2x V100, top_k 1, cache_prompt false) 19 of 20 completions are
    // byte-identical and one -- an arithmetic word problem -- diverges partway through:
    //   FUGSPLIT=0  82b9c8dcfd319215a9a0bdf53f41eec3ce9232ea223213dae87d61d6f3867ecb
    //   FUGSPLIT=1  df8fe4db9903fdb5d07f1adcaad1aefefaf753335966714545933662ff6125ea
    // The hash gate that note asked for has now been run and it does NOT pass, so this is a
    // numerics-affecting lever, not a free one. It is defaulted on for the decode win with that
    // understood; PXA_PXQ_MMVQ_FUGSPLIT=0 restores the fused twin exactly.
    const int cc_fug = ggml_cuda_info().devices[ctx.device].cc;
    static const int fug_env = [](){
        const char * e = getenv("PXA_PXQ_MMVQ_FUGSPLIT");
        return e ? atoi(e) : -1;
    }();
    const bool pxq_fugsplit = fug_env >= 0 ? (fug_env != 0) : (cc_fug >= CC_VOLTA);
    {
        static std::atomic<uint32_t> fug_seen{0};
        const uint32_t fug_bit = 1u << (2*(ctx.device & 15) + (pxq_fugsplit ? 0 : 1));
        if (!(fug_seen.fetch_or(fug_bit) & fug_bit)) {
            fprintf(stderr, "PXA_PXQ_MMVQ_FUGSPLIT dev%d: %s (dense PXQ up/gate decode = %s)%s\n",
                    ctx.device, pxq_fugsplit ? "ON" : "OFF",
                    pxq_fugsplit ? "2x plain MMVQ + GLU epilogue" : "fused twin",
                    fug_env >= 0 ? " [explicit PXA_PXQ_MMVQ_FUGSPLIT]" : " [auto: sm_70+]");
        }
    }
    if (pxq_mmvq_fug && pxq_fugsplit) {
        // limit sanitize EXACTLY as ggml_cuda_op_mul_mat_vec_q_impl does (limit<=1e-6 -> INF):
        // with INF the elementwise SILU takes its limit variant, whose op order matches the
        // fused kernel epilogue bit-for-bit (min/max with INF are identities). The raw 0.0
        // would select the no-limit variant, (g*u)/(1+e) vs the fused u*(g/(1+e)) -- one
        // rounding difference per element, caught by the hash gate on 2026-08-10.
        float limit_fs = *(const float *)(dst->op_params + 1);
        limit_fs = limit_fs > 1e-6f ? limit_fs : INFINITY;
        const ggml_unary_op uop_fs = (ggml_unary_op)dst->op_params[0];
        ggml_cuda_pool_alloc<float> dst_up_fs(ctx.pool(), ggml_nelements(dst));
        auto local_dst = *dst;
        local_dst.data = dst_up_fs.get();
        local_dst.op_params[0] = GGML_PREC_DEFAULT;
        auto gate_dst = *dst;                 // same data pointer: the result still lands in dst
        gate_dst.op_params[0] = GGML_PREC_DEFAULT;
        ggml_cuda_mul_mat(ctx, src0_1, src1, &local_dst, nullptr, 0);
        ggml_cuda_mul_mat(ctx, src0_2, src1, &gate_dst, nullptr, 0);
        ggml_fused_mul_unary(ctx, uop_fs, ggml_nelements(dst),
                        (const float *)dst->data, dst_up_fs.get(), (float *)dst->data, limit_fs);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (!pxq_mmvq_fug && (pxa_is_pxq_type(src0_1->type) || pxa_is_pxq_type(src0_2->type))) {
        // Fused single-launch decode form first; it declines to the split path below for any
        // shape/op/geometry it cannot serve.
        if (pxa_pxq_gateup_2d(ctx, dst) == 0) return;
        const float limit_px = *(const float *)(dst->op_params + 1);
        // PXA_DENSE_FUG_PREC_v1: op_params[0] on a FUSED_UP_GATE node is the UNARY OP id
        // (GGML_UNARY_OP_SILU == 10), NOT a ggml_prec. Both ggml_cuda_op_mul_mat_cublas (which
        // gates cublasGemmEx CUDA_R_16F + TENSOR_OP on op_params[0] == GGML_PREC_DEFAULT) and
        // pxa_pxq_gemm_2d read it as a precision, so the alias vetoed fp16 on BOTH the up and the
        // gate GEMM and dropped them to dequant-to-f32 + cublasSgemm. On sm_60 that costs the 2:1
        // hgemm; on sm_70 it costs the tensor cores outright (volta_sgemm ~15.7 TFLOPS vs HMMA
        // ~112). Same defect, same fix as PXA_P100_FP16_GEMM_v1 on the MoE path -- that one only
        // ever cleared the alias for MUL_MAT_ID, and the dense node was left leaking.
        // Read the op and the limit from the ORIGINAL dst, then hand both GEMMs clean copies.
        const ggml_unary_op uop_px = (ggml_unary_op)dst->op_params[0];
        ggml_cuda_pool_alloc<float> dst_up_px(ctx.pool(), ggml_nelements(dst));
        auto local_dst = *dst;
        local_dst.data = dst_up_px.get();
        local_dst.op_params[0] = GGML_PREC_DEFAULT;
        auto gate_dst = *dst;                 // same data pointer: the result still lands in dst
        gate_dst.op_params[0] = GGML_PREC_DEFAULT;
        ggml_cuda_mul_mat(ctx, src0_1, src1, &local_dst, nullptr, 0);
        ggml_cuda_mul_mat(ctx, src0_2, src1, &gate_dst, nullptr, 0);
        ggml_fused_mul_unary(ctx, uop_px, ggml_nelements(dst),
                        (const float *)dst->data, dst_up_px.get(), (float *)dst->data, limit_px);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    GGML_ASSERT(ggml_is_quantized(src0_1->type));
    GGML_ASSERT(src0_1->type == src0_2->type);
    GGML_ASSERT(src1->ne[2] == 1);
    GGML_ASSERT(src1->ne[3] == 1);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    auto stream = ctx.stream();

    float limit = *(const float *)(dst->op_params + 1);

    auto ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    auto nb10_padded = ne10_padded*sizeof(block_q8_1)/QK8_1;
    auto quantized_size = nb10_padded*src1->ne[1];
    if (src1->ne[1] > 8) {
        quantized_size += get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
    }
    ggml_cuda_pool_alloc<float> dst_up(ctx.pool(), ggml_nelements(dst));
    ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), quantized_size);
    if (src1->ne[1] <= 8) {
        // G2-F3: reuse a fused-norm q8_1 sidecar of src1 if one is valid (skips the quantize)
        const char * q8u = pxa_g2_q8_lookup(ctx.device, src1, ne10_padded);
        if (!q8u) {
            quantize_row_q8_1_cuda((const float *)src1->data, (void *)src1_quantized.get(), src1->ne[0], src1->ne[1], 1, ne10_padded,
                    src0_1->type, stream);
            CUDA_CHECK(cudaGetLastError());
            q8u = src1_quantized.get();
        }

        if (src1->ne[1] == 1 && src0_1->type == src0_2->type) {
            ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, src0_1, src1, nullptr, dst,
                    dst->src[4], dst->src[5],
                    (const char *)src0_1->data, (const char *)src0_2->data, (const float *)src1->data, q8u,
                    (float *)dst->data, 0, src0_1->ne[1], 1, ne10_padded,
                    (ggml_unary_op)dst->op_params[0], limit, stream);
            return;
        }

        ggml_cuda_op_mul_mat_vec_q(ctx, src0_1, src1, dst, (const char *)src0_1->data, nullptr, q8u, dst_up.get(),
                0, src0_1->ne[1], src1->ne[1], ne10_padded, stream);
        CUDA_CHECK(cudaGetLastError());

        ggml_cuda_op_mul_mat_vec_q(ctx, src0_2, src1, dst, (const char *)src0_2->data, nullptr, q8u, (float *)dst->data,
                0, src0_2->ne[1], src1->ne[1], ne10_padded, stream);
        CUDA_CHECK(cudaGetLastError());
    } else {

        if (ggml_cuda_should_use_mmq(src0_1->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[1])) {
            quantize_mmq_q8_1_cuda((const float *)src1->data, src1_quantized.get(), src1->ne[0], src1->ne[1], 1,
                    ne10_padded, src0_1->type, stream);
            CUDA_CHECK(cudaGetLastError());

            ggml_cuda_op_mul_mat_q(ctx, src0_1, src1, dst, (const char *)src0_1->data, nullptr, src1_quantized.get(), dst_up.get(),
                    0, src0_1->ne[1], src1->ne[1], ne10_padded, stream);
            CUDA_CHECK(cudaGetLastError());

            ggml_cuda_op_mul_mat_q(ctx, src0_2, src1, dst, (const char *)src0_2->data, nullptr, src1_quantized.get(), (float *)dst->data,
                    0, src0_2->ne[1], src1->ne[1], ne10_padded, stream);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto local_dst = *dst;
            local_dst.data = dst_up.get();
            // PXA_DENSE_FUG_PREC_KQ (2026-08-10, default OFF): THIRD instance of the
            // op_params[0] alias defect. On a FUSED_UP_GATE node op_params[0] is the UNARY
            // OP id (SILU == 10), not a ggml_prec; ggml_cuda_op_mul_mat_cublas reads it as
            // a precision and vetoes the fp16 GemmEx branch -> dequant-to-FP32 + volta_sgemm
            // (20.1 ms/call measured, 50.7% of the Muse Q4_K_XL prefill wall on 2x V100).
            // The PXQ dense branch above sanitizes its dst copies (PXA_DENSE_FUG_PREC_v1);
            // this generic k-quant branch never got the fix. Armed: hand both GEMMs clean
            // PREC_DEFAULT copies, exactly as the PXQ branch does. G3-class numerics
            // (fp16-accumulate h884 vs the fp32 SGEMM the alias forced).
            static const bool pxa_fug_prec_kq = [](){
                const char * e = getenv("PXA_DENSE_FUG_PREC_KQ");
                const bool v = e && atoi(e) != 0;
                if (v) fprintf(stderr, "PXA_DENSE_FUG_PREC_KQ ARMED: generic dense up/gate GEMMs run with sanitized PREC_DEFAULT (fp16 GemmEx path)\n");
                return v;
            }();
            auto gate_dst = *dst;
            if (pxa_fug_prec_kq) {
                local_dst.op_params[0] = GGML_PREC_DEFAULT;
                gate_dst.op_params[0]  = GGML_PREC_DEFAULT;
            }
            ggml_cuda_mul_mat(ctx, src0_1, src1, &local_dst, nullptr, 0);
            ggml_cuda_mul_mat(ctx, src0_2, src1, pxa_fug_prec_kq ? &gate_dst : dst, nullptr, 0);
        }
    }

    //printf("%s: using limit = %g\n", __func__, limit);
    ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(dst),
                    (const float *)dst->data, dst_up.get(), (float *)dst->data, limit);
    CUDA_CHECK(cudaGetLastError());

}

static inline bool ops_are_same_device(const ggml_cgraph * cgraph, int first, int last) {
    if (last <= first) return true;
    int device = ((const ggml_backend_cuda_buffer_context *)cgraph->nodes[first]->buffer->context)->device;
    for (int i = first; i <= last; ++i) {
        auto node = cgraph->nodes[i];
        if (((const ggml_backend_cuda_buffer_context *)node->buffer->context)->device != device) return false;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (!node->src[j] || !node->src[j]->buffer) continue;
            if (((const ggml_backend_cuda_buffer_context *)node->src[j]->buffer->context)->device != device) return false;
        }
    }
    return true;
}

// G2-F3: does the FUSED_RMS_NORM output `dst` feed at least one q8_1-quantized GEMV consumer
// (a quantized mmvq MUL_MAT, or a FUSED_UP_GATE, on this device) within the lookahead window?
// If so, report the consumer's ne10_padded so the sidecar is laid out to match.
static bool pxa_g2_normfuse_wanted(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i,
        const ggml_tensor * dst, int64_t & padded_out) {
    if (!ggml_is_contiguous(dst)) return false;
    if (ggml_nelements(dst) > 65536) return false;               // decode-shape only
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 20; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        const ggml_tensor * s1 = nullptr;
        if (n->op == GGML_OP_MUL_MAT)            s1 = n->src[1];
        else if (n->op == GGML_OP_FUSED_UP_GATE) s1 = n->src[2];
        else continue;
        if (!s1) continue;
        const ggml_tensor * base = s1->view_src ? s1->view_src : s1;
        if (base != dst || s1->data != dst->data) continue;
        if (ggml_nrows(s1) != 1 || !ggml_is_contiguous(s1)) continue;
        if (ggml_nelements(s1) != ggml_nelements(dst)) continue;
        const ggml_tensor * w = n->src[0];
        if (!w || !ggml_is_quantized(w->type) || !ggml_cuda_mmvq_type_supported(w->type)) continue;
        if (!n->buffer || !ggml_backend_buffer_is_cuda(n->buffer)) continue;
        if (((ggml_backend_cuda_buffer_context *)n->buffer->context)->device != ctx.device) continue;
        padded_out = GGML_PAD(s1->ne[0], MATRIX_ROW_PADDING);
        if (ggml_nrows(dst) > 1 && padded_out != ggml_nelements(dst)) continue;  // flat emit needs pad-free layout
        return true;
    }
    return false;
}

// G2-F4: does tensor t have any consumer in nodes (i+1..end) besides `except`?
// G2-F4a WAR SAFETY (2026-09-02, determinism defect on the published PXQ2 tier).
//
// Fusing the ADD into FUSED_RMS_NORM removes the kernel boundary that made the pair safe.
// Unfused, the launch boundary is a grid-wide barrier: every read of the ADD's two inputs has
// completed before the norm kernel writes a single byte of its own output. Fused, the only
// barrier is the per-block __syncthreads() inside the reduction, which orders nothing between
// blocks -- and the norm's output is exactly the buffer the allocator is most likely to place
// on top of an ADD input, because BOTH ADD inputs die at the ADD (the residual's last use is
// this add, the attention output's only use is this add), so their blocks are free and get
// merged with their neighbours before the norm output is placed. When the merge shifts the
// base, dst[row*ncols+col] and a/b[row*ncols+col] are the same memory at DIFFERENT indices, so
// the epilogue store of one row's block lands in another row's not-yet-read input: a
// write-after-read race whose outcome depends on block scheduling.
//
// Measured: on the published PXA-Fusion2-35B PXQ2 tier at 5,801 tokens this made every greedy
// temp-0 completion differ; the first divergence is the fused node itself (both of its inputs
// bit-identical across reps, its output not).
//
// Exact aliasing is harmless -- each block reads and writes only its own row, so dst == a or
// dst == b is index-for-index -- so only a SHIFTED overlap declines, and the fusion keeps
// firing everywhere it was already safe.
static inline bool pxa_g2_addfuse_no_shifted_overlap(const ggml_tensor * d, const ggml_tensor * s) {
    if (!d->data || !s->data)   return true;
    if (d->buffer != s->buffer) return true;   // separate allocations cannot overlap
    if (d->data == s->data)     return true;   // exact alias: every access is index-for-index
    const char * d0 = (const char *) d->data, * d1 = d0 + ggml_nbytes(d);
    const char * s0 = (const char *) s->data, * s1 = s0 + ggml_nbytes(s);
    return d1 <= s0 || s1 <= d0;               // disjoint
}

// PXA_G2_FUSE_WAR=0 restores the unguarded G2-F5 fusions for A/B measurement only. Same
// hazard and same test as the ADD/RMS guard above: every one of these fused kernels is
// elementwise, thread i reads index i and writes index i, so an EXACT alias is index-for-index
// safe and only a SHIFTED overlap between the fused output and any input it no longer has a
// kernel boundary against can race.
static inline bool pxa_g2_fuse_war() {
    static const bool on = [](){
        const char * e = getenv("PXA_G2_FUSE_WAR");
        const bool v = e ? atoi(e) != 0 : true;
        if (!v) fprintf(stderr, "PXA_G2_FUSE_WAR: OFF -- G2-F5 fusions are UNGUARDED "
                        "(known write-after-read race; measurement only, never ship)\n");
        return v;
    }();
    return on;
}

// PXA_G2_ADDFUSE_WAR=0 restores the unguarded (racy) fusion for A/B measurement only.
static inline bool pxa_g2_addfuse_war() {
    static const bool on = [](){
        const char * e = getenv("PXA_G2_ADDFUSE_WAR");
        const bool v = e ? atoi(e) != 0 : true;
        if (!v) fprintf(stderr, "PXA_G2_ADDFUSE_WAR: OFF -- ADD+FUSED_RMS_NORM fusion is UNGUARDED "
                        "(known write-after-read race; measurement only, never ship)\n");
        return v;
    }();
    return on;
}

static inline void pxa_g2_addfuse_dbg(const ggml_tensor * norm, const ggml_tensor * a,
                                      const ggml_tensor * b, bool fired) {
    static const bool on = getenv("PXA_G2_ADDFUSE_DBG") != nullptr;
    if (!on) return;
    static std::atomic<int> left{40};
    if (left.fetch_sub(1) <= 0) return;
    auto delta = [](const ggml_tensor * d, const ggml_tensor * s) -> long long {
        if (d->buffer != s->buffer || !d->data || !s->data) return (long long)1e18;
        return (long long)((const char *)d->data - (const char *)s->data);
    };
    fprintf(stderr, "PXA_G2_ADDFUSE_DBG: %-24s %s  norm=%p a=%p b=%p  d(a)=%lld d(b)=%lld "
            "nb_norm=%zu nb_a=%zu nb_b=%zu\n", norm->name, fired ? "FUSE " : "DECLINE",
            norm->data, a->data, b->data, delta(norm, a), delta(norm, b),
            ggml_nbytes(norm), ggml_nbytes(a), ggml_nbytes(b));
}

static bool pxa_g2_sole_consumer(const ggml_cgraph * cgraph, int i, const ggml_tensor * t, const ggml_tensor * except) {
    for (int j = i + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n == except) continue;
        if (n->view_src == t) return false;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] == t) return false;
        }
    }
    return true;
}


#include "ggml-cuda/pxa/pxa-deltanet-fuse.cuh"
#include "ggml-cuda/pxa/pxa-ew-fuse.cuh"

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst, const ggml_cgraph * cgraph, int & i) {

#if IK_PRINT_TIMING
    int64_t tim1 = ggml_time_us();
#endif

    if (ggml_is_noop(dst)) {
        return true;
    }

    // In case we forget to do that in some kernel.
    ggml_cuda_set_device(ctx.device);

    // PXA_X_CACHE validity clock: one tick per EXECUTED graph node. The activation-stage cache is
    // only honoured when its conversion happened on the immediately preceding tick, which is what
    // makes "nothing wrote through an alias of src1 in between" true rather than hoped for.
    pxa_op_seq.fetch_add(1, std::memory_order_relaxed);

    auto next = i < cgraph->n_nodes - 1 ? cgraph->nodes[i+1] : nullptr;

    auto fusion = ctx.fusion;

    // PXA_EW_FUSE: collapse a straight-line elementwise run starting here into one kernel
    // (bit-identical; see pxa-ew-fuse.cuh). Declines (env off, chain < min, any guard) fall
    // through to the per-op switch unchanged.
    if (fusion && pxa_ew_fuse_enabled()) {
        const int pxa_ew_n = pxa_try_ew_chain(ctx, cgraph, i);
        if (pxa_ew_n > 0) {
            i += pxa_ew_n - 1;
            return true;
        }
    }

    //printf("%4d %s(%s) on device %d. time = %ld\n", i, ggml_op_name(dst->op), dst->name, ctx.device, ggml_time_us());
    switch (dst->op) {
        case GGML_OP_REDUCE:
            ggml_cuda_op_reduce(ctx, dst);
            break;
        case GGML_OP_FAKE_CPY:
            break;
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_HADAMARD:
            ggml_cuda_op_hadamard(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            // PXA_FUSE_DELTANET bit2: the delta-net safe-state-path gather + reset-mask multiply
            if (fusion) {
                const int pxa_gm = pxa_try_deltanet_gather_mask(ctx, cgraph, i);
                if (pxa_gm > 0) { i += pxa_gm; break; }
            }
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            if (pxa_g2_cpy2fuse() && fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_CPY &&
                ops_are_same_device(cgraph, i, i+2) &&
                // G2-F5c: unfused, the first copy completes before the second starts, so a
                // shifted overlap between one copy's destination and the OTHER copy's source
                // is safe. Fused into one kernel it is a write-after-read race.
                (!pxa_g2_fuse_war() ||
                 (pxa_g2_addfuse_no_shifted_overlap(dst->src[1], cgraph->nodes[i+2]->src[0]) &&
                  pxa_g2_addfuse_no_shifted_overlap(cgraph->nodes[i+2]->src[1], dst->src[0]) &&
                  pxa_g2_addfuse_no_shifted_overlap(dst->src[1], cgraph->nodes[i+2]->src[1]))) &&
                ggml_cuda_cpy_2(ctx, dst->src[0], cgraph->nodes[i+2]->src[0], dst->src[1], cgraph->nodes[i+2]->src[1])) {
                i += 2;
            } else {
                ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            }
            break;
        case GGML_OP_CONT:
            if (pxa_g2_sumrowsfuse() && fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_SUM_ROWS &&
                cgraph->nodes[i+2]->op == GGML_OP_TRANSPOSE &&
                dst->src[0]->op == GGML_OP_TRANSPOSE &&
                ops_are_same_device(cgraph, i, i+2) &&
                // G2-F5d: dropping the CONT makes the reduction read the pre-cont source
                // directly, so the reduce output must not sit on top of it.
                (!pxa_g2_fuse_war() ||
                 pxa_g2_addfuse_no_shifted_overlap(cgraph->nodes[i+1], dst->src[0])) ) {
                ggml_cuda_op_sum_rows_nc(ctx, cgraph->nodes[i+1]);
                i += 2;
            } else {
                ggml_cuda_dup(ctx, dst);
            }
            break;
        case GGML_OP_ADD:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_ADD &&
                cgraph->nodes[i+2]->op == GGML_OP_FUSED_RMS_NORM &&
                ggml_is_contiguous(dst->src[0]) &&
                ggml_is_contiguous(dst->src[1]) &&
                dst->src[0]->type == GGML_TYPE_F32 &&               // with split mode "attn" we can end up having f16
                ggml_are_same_shape(dst->src[0], dst->src[1]) &&
                dst == cgraph->nodes[i+1]->src[0] &&
                ggml_is_contiguous(cgraph->nodes[i+1]->src[1]) &&
                ggml_are_same_shape(dst, cgraph->nodes[i+1]->src[1]) &&
                cgraph->nodes[i+1] == cgraph->nodes[i+2]->src[0] &&
                ops_are_same_device(cgraph, i, i+2) &&
                // same write-after-read guard as the ADD+FUSED_RMS_NORM pair below: fusing away
                // the kernel boundary lets this kernel's norm-output stores land on a buffer
                // another block has not finished reading as an ADD input.
                (!pxa_g2_addfuse_war() ||
                 (pxa_g2_addfuse_no_shifted_overlap(cgraph->nodes[i+2], dst->src[0]) &&
                  pxa_g2_addfuse_no_shifted_overlap(cgraph->nodes[i+2], dst->src[1]) &&
                  pxa_g2_addfuse_no_shifted_overlap(cgraph->nodes[i+2], cgraph->nodes[i+1]->src[1])))) {
                ggml_cuda_op_fused_add_add_rms_norm(ctx, dst, cgraph->nodes[i+1], cgraph->nodes[i+2]);
                i += 2;
            }
            else if (pxa_g2_softplusfuse() && fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_UNARY &&
                (ggml_unary_op)cgraph->nodes[i+1]->op_params[0] == GGML_UNARY_OP_SOFTPLUS &&
                cgraph->nodes[i+2]->op == GGML_OP_MUL &&
                cgraph->nodes[i+2]->src[0] == cgraph->nodes[i+1] &&
                cgraph->nodes[i+1]->src[0] == cgraph->nodes[i] &&
                ggml_nrows(cgraph->nodes[i+0]->src[1]) == 1 &&
                ggml_nrows(cgraph->nodes[i+2]->src[1]) == 1 &&
                ops_are_same_device(cgraph, i, i+2) &&
                // G2-F5a: this is the delta-net gate, alpha_biased -> softplus -> gate. The
                // fused kernel writes the MUL output while other blocks are still reading the
                // ADD inputs, so a shifted alias corrupts the gate and, through the
                // recurrence, the recurrent state itself.
                (!pxa_g2_fuse_war() ||
                 (pxa_g2_addfuse_no_shifted_overlap(cgraph->nodes[i+2], dst->src[0]) &&
                  pxa_g2_addfuse_no_shifted_overlap(cgraph->nodes[i+2], dst->src[1]) &&
                  pxa_g2_addfuse_no_shifted_overlap(cgraph->nodes[i+2], cgraph->nodes[i+2]->src[1])))) {
                ggml_cuda_fused_softplus(ctx, cgraph->nodes[i+2]);
                i += 2;
            }
            // G2-F4 ADDFUSE (2026-07-19): re-enable the upstream ADD+FUSED_RMS_NORM pair fusion,
            // env-gated. ne0 >= 256 keeps the fused kernel's block-size selection (256/1024)
            // identical to the standalone fused_rms_norm launcher's, so the reduction order and
            // therefore the output are bit-identical to k_add_same + fused_rms_norm_f32.
            else if (pxa_g2_addfuse() && pxa_g2_addfuse_rms() && fusion && i + 1 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_FUSED_RMS_NORM &&
                ggml_is_contiguous(dst->src[0]) &&
                ggml_is_contiguous(dst->src[1]) &&
                dst->src[0]->type == GGML_TYPE_F32 &&
                dst->src[1]->type == GGML_TYPE_F32 &&
                dst->ne[0] >= 256 &&
                ggml_are_same_shape(dst->src[0], dst->src[1]) &&
                cgraph->nodes[i+1]->src[1]->type == GGML_TYPE_F32 &&
                ggml_nrows(cgraph->nodes[i+1]->src[1]) == 1 &&
                dst == cgraph->nodes[i+1]->src[0] && ops_are_same_device(cgraph, i, i+1) &&
                (!pxa_g2_addfuse_war() ||
                 (pxa_g2_addfuse_no_shifted_overlap(cgraph->nodes[i+1], dst->src[0]) &&
                  pxa_g2_addfuse_no_shifted_overlap(cgraph->nodes[i+1], dst->src[1])))) {
                pxa_g2_addfuse_dbg(cgraph->nodes[i+1], dst->src[0], dst->src[1], true);
                ggml_cuda_op_fused_add_rms_norm(ctx, dst, cgraph->nodes[i+1]);
                ++i;
            } else {
                if (pxa_g2_addfuse() && pxa_g2_addfuse_rms() && fusion && i + 1 < cgraph->n_nodes &&
                    cgraph->nodes[i+1]->op == GGML_OP_FUSED_RMS_NORM && dst == cgraph->nodes[i+1]->src[0]) {
                    pxa_g2_addfuse_dbg(cgraph->nodes[i+1], dst->src[0], dst->src[1], false);
                }
                ggml_cuda_op_add(ctx, dst);
            }
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_MULTI_ADD:
            ggml_cuda_op_multi_add(ctx, dst);
            break;
        case GGML_OP_MUL_MULTI_ADD:
            // G2-F4 ADDFUSE: fold the following residual ADD into the mul_multi_add epilogue
            // (experts summed first, residual added last => bit-identical in either operand
            // order). Requires the multi-add output to have NO consumer besides that ADD.
            if (pxa_g2_addfuse() && pxa_g2_addfuse_mma() && fusion && next && next->op == GGML_OP_ADD &&
                (next->src[0] == dst || next->src[1] == dst) &&
                next->type == GGML_TYPE_F32 &&
                (next->src[0] == dst ? next->src[1] : next->src[0])->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(next->src[0], next->src[1]) &&
                ggml_are_same_shape(next, dst) &&
                ggml_is_contiguous(next->src[0]) && ggml_is_contiguous(next->src[1]) &&
                ops_are_same_device(cgraph, i, i+1) &&
                pxa_g2_sole_consumer(cgraph, i, dst, next)) {
                ggml_cuda_op_mul_multi_add_fused(ctx, dst, next);
                ++i;
            } else {
                ggml_cuda_op_mul_multi_add(ctx, dst);
            }
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            if (pxa_g2_mulexpfuse() && fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_UNARY &&
                cgraph->nodes[i+2]->op == GGML_OP_MUL &&
                (ggml_unary_op)cgraph->nodes[i+1]->op_params[0] == GGML_UNARY_OP_EXP &&
                cgraph->nodes[i+1]->src[0] == dst &&
                cgraph->nodes[i+2]->src[0] == cgraph->nodes[i+1] &&
                cgraph->nodes[i+2]->src[1] == dst->src[1] &&
                ops_are_same_device(cgraph, i, i+2) &&
                // G2-F5b: same hazard, on the delta-net decay term.
                (!pxa_g2_fuse_war() ||
                 (pxa_g2_addfuse_no_shifted_overlap(cgraph->nodes[i+2], dst->src[0]) &&
                  pxa_g2_addfuse_no_shifted_overlap(cgraph->nodes[i+2], dst->src[1]))) ) {
                ggml_cuda_fused_mul_exp_mul(ctx, cgraph->nodes[i+2]);
                i += 2;
                //printf("mul(%s) -> exp(%s) -> mul(%s), %d, %d, %zu, %zu; %ld x %ld x %ld x %ld - %ld x %ld x %ld x %ld\n", dst->name, cgraph->nodes[i+1]->name, cgraph->nodes[i+2]->name,
                //        ggml_is_contiguous(dst->src[0]), ggml_is_contiguous(dst->src[1]), ggml_nelements(dst->src[0]), ggml_nelements(dst->src[1]),
                //        dst->src[0]->ne[0], dst->src[0]->ne[1], dst->src[0]->ne[2], dst->src[0]->ne[3],
                //        dst->src[1]->ne[0], dst->src[1]->ne[1], dst->src[1]->ne[2], dst->src[1]->ne[3]);
            } else {
                //printf("mul(%s): %d, %d, %d, %ld x %ld x %ld x %ld * %ld x %ld x %ld x %ld\n", dst->name, ggml_is_contiguous(dst->src[0]), ggml_is_contiguous(dst->src[1]), ggml_is_contiguous(dst),
                //        dst->src[0]->ne[0], dst->src[0]->ne[1], dst->src[0]->ne[2], dst->src[0]->ne[3],
                //        dst->src[1]->ne[0], dst->src[1]->ne[1], dst->src[1]->ne[2], dst->src[1]->ne[3]);
                ggml_cuda_op_mul(ctx, dst);
            }
            break;
        case GGML_OP_FUSED_MUL_UNARY:
            // PXA_SHEXP_GATE_FUSE consumer slot: fires the deferred gate GEMV fused with
            // this node's sigmoid-mul. Off / no pending -> stock path, zero cost.
            if (!pxa_shexp_gate_fire(ctx, dst)) {
                ggml_cuda_op_fused_mul_unary(ctx, dst);
            }
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_UNARY:
            //printf("unary(%s, %s)\n", dst->name, ggml_unary_op_name((ggml_unary_op)dst->op_params[0]));
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    if (fusion) {
                        // PXA_FUSE_DELTANET bit0: the DeltaNet decode qk-norm/state-writeback cluster
                        int pxa_n = pxa_try_deltanet_cluster(ctx, cgraph, i);
                        if (pxa_n > 0) { i += pxa_n; break; }
                    }
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                // PXA: ABS and SGN had working CUDA kernels in unary.cu but were
                // never registered here or in supports_op, so the scheduler placed
                // them on the CPU. In qwen4exp that is not cosmetic: the PLE gate
                // computes sigmoid(sgn(x) * sqrt(clamp(abs(x), eps, inf))), so two
                // 8 KB unaries dragged a whole D2H/H2D round trip into the middle of
                // layer 1 and shattered the CUDA0 graph segment into extra splits.
                // Both ops are exactly representable, so this is bit-identical.
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    if (fusion && i + 5 < cgraph->n_nodes &&
                        cgraph->nodes[i+1]->op == GGML_OP_RESHAPE &&
                        cgraph->nodes[i+2]->op == GGML_OP_ADD &&
                        cgraph->nodes[i+3]->op == GGML_OP_ARGSORT &&
                        cgraph->nodes[i+4]->op == GGML_OP_VIEW &&
                        cgraph->nodes[i+5]->op == GGML_OP_GET_ROWS && ops_are_same_device(cgraph, i, i+5)) {
                        cuda_glm45moe_experts(ctx, cgraph->nodes[i+5], cgraph->nodes[i+4]);
                        i += 5;
                    }
                    else if (fusion && i + 4 < cgraph->n_nodes &&
                        cgraph->nodes[i+1]->op == GGML_OP_RESHAPE &&
                        cgraph->nodes[i+2]->op == GGML_OP_ADD &&
                        cgraph->nodes[i+3]->op == GGML_OP_GROUPED_TOPK &&
                        cgraph->nodes[i+4]->op == GGML_OP_GET_ROWS && ops_are_same_device(cgraph, i, i+4)) {
                        cuda_bailingmoev2_experts(ctx, cgraph->nodes[i+4], cgraph->nodes[i+3]);
                        i += 4;
                    } else if (fusion && i + 2 < cgraph->n_nodes &&
                        cgraph->nodes[i+1]->op == GGML_OP_RESHAPE &&
                        cgraph->nodes[i+2]->op == GGML_OP_ADD && ops_are_same_device(cgraph, i, i+2)) {
                        ggml_cuda_op_biased_sigmoid(ctx, cgraph->nodes[i+2]);
                        i += 2;
                    } else {
                        ggml_cuda_op_sigmoid(ctx, dst);
                    }
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return -1;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_CPY &&
                ggml_cuda_concat_cpy(ctx, dst, cgraph->nodes[i+2])) {
                i += 2;
            } else {
                ggml_cuda_op_concat(ctx, dst);
            }
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_FUSED_RMS_NORM:
            if (fusion && pxa_try_deltanet_outgate(ctx, cgraph, i)) {
                // PXA_FUSE_DELTANET bit1: DeltaNet out-gate rms+silu fusion
                i += 1;
            }
            else if (false && fusion && i + 4 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_FUSED_RMS_NORM &&
                cgraph->nodes[i+3]->op == GGML_OP_ROPE_FAST &&
                cgraph->nodes[i+4]->op == GGML_OP_ROPE_FAST &&
                ggml_cuda_op_fused_rms_rope_fast(ctx, cgraph->nodes[i+3], cgraph->nodes[i+4])) {
                i += 4;
            }
            else if (false && fusion && i + 4 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_ROPE_FAST &&
                cgraph->nodes[i+2]->op == GGML_OP_RESHAPE &&
                cgraph->nodes[i+3]->op == GGML_OP_FUSED_RMS_NORM &&
                cgraph->nodes[i+4]->op == GGML_OP_ROPE_FAST &&
                ggml_cuda_op_fused_rms_rope_fast(ctx, cgraph->nodes[i+1], cgraph->nodes[i+4])) {
                i += 4;
            }
            else if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_FUSED_RMS_NORM &&
                dst->ne[2] == 1 && cgraph->nodes[i+2]->ne[2] == 1) {
                ggml_cuda_op_fused_rms_rms_norm(ctx, dst, cgraph->nodes[i+2]);
                i += 2;
            } else {
                // G2-F3 NORMFUSE: emit the q8_1 sidecar alongside the norm when a quantized
                // GEMV consumer is coming (f32 dst stays bit-identical; miss -> plain path).
                bool g2_done = false;
                int64_t g2_padded = 0;
                if (pxa_g2_normfuse() && fusion && dst->src[1] && ggml_nrows(dst) == 1 &&
                    pxa_g2_normfuse_wanted(ctx, cgraph, i, dst, g2_padded)) {
                    const size_t need = (size_t)(g2_padded/QK8_1)*sizeof(block_q8_1);
                    char * q8 = pxa_g2_q8_buf(ctx.device, ctx.stream(), need);
                    if (q8 && ggml_cuda_op_fused_rms_norm_q8(ctx, dst, q8, g2_padded)) {
                        auto & sc = pxa_g2_q8sc[ctx.device];
                        sc.t = dst; sc.data = dst->data; sc.padded = g2_padded; sc.eval = pxa_g2_eval_serial;
                        g2_done = true;
                    }
                }
                if (!g2_done) {
                    ggml_cuda_op_fused_rms_norm(ctx, dst);
                }
            }
            break;
        case GGML_OP_FUSED_RMS_RMS_ADD:
            ggml_cuda_op_fused_rms_rms_add(ctx, dst);
            break;
        case GGML_OP_FUSED_NORM:
            ggml_cuda_op_fused_rms_norm(ctx, dst, true);
            break;
        case GGML_OP_MUL_MAT:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                GGML_CUDA_LOG_ERROR("%s: cannot compute %s: src0->ne[3] = %" PRId64 ", src1->ne[3] = %" PRId64 " - fallback to CPU\n", __func__, dst->name, dst->src[0]->ne[3], dst->src[1]->ne[3]);
                return -1;
            } else {
                i = ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst, cgraph, i);
            }
            break;
        case GGML_OP_MUL_MAT_ID:
            if (ggml_cuda_mul_mat_id(ctx, dst, next)) ++i;
            break;
        case GGML_OP_MOE_FUSED_UP_GATE:
            i = ggml_cuda_moe_up_gate_unary(ctx, dst, cgraph, i);
            break;
        case GGML_OP_FUSED_UP_GATE:
            ggml_cuda_up_gate_unary(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SOFTCAP:
            ggml_cuda_op_softcap(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            if (fusion && i + 8 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_RESHAPE  &&
                cgraph->nodes[i+2]->op == GGML_OP_ADD &&
                cgraph->nodes[i+3]->op == GGML_OP_ARGSORT &&
                cgraph->nodes[i+4]->op == GGML_OP_VIEW     &&
                cgraph->nodes[i+5]->op == GGML_OP_GET_ROWS &&
                cgraph->nodes[i+6]->op == GGML_OP_RESHAPE  &&
                cgraph->nodes[i+7]->op == GGML_OP_SUM_ROWS &&
                cgraph->nodes[i+8]->op == GGML_OP_DIV) {
                ggml_cuda_op_topk_moe(ctx, cgraph->nodes[i], cgraph->nodes[i+8], cgraph->nodes[i+4], cgraph->nodes[i+2]->src[1]);
                pxa_glue_topk_note("softmax+bias+norm", dst);
                i += 8;
            }
            else if (fusion && i + 4 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_RESHAPE  &&
                cgraph->nodes[i+2]->op == GGML_OP_ARGSORT  &&
                cgraph->nodes[i+3]->op == GGML_OP_VIEW     &&
                cgraph->nodes[i+4]->op == GGML_OP_GET_ROWS &&
                ggml_cuda_should_use_topk_moe(cgraph->nodes[i], cgraph->nodes[i+4]) &&
                ops_are_same_device(cgraph, i, i+4)) {
                if (i + 7 < cgraph->n_nodes &&
                    cgraph->nodes[i+5]->op == GGML_OP_RESHAPE  &&
                    cgraph->nodes[i+6]->op == GGML_OP_SUM_ROWS &&
                    cgraph->nodes[i+7]->op == GGML_OP_DIV) {
                    ggml_cuda_op_topk_moe(ctx, cgraph->nodes[i], cgraph->nodes[i+7], cgraph->nodes[i+3]);
                    pxa_glue_topk_note("softmax+norm", dst);
                    i += 7;
                } else {
                    ggml_cuda_op_topk_moe(ctx, cgraph->nodes[i], cgraph->nodes[i+4], cgraph->nodes[i+3]);
                    pxa_glue_topk_note("softmax", dst);
                    i += 4;
                }
            } else {
                pxa_glue_topk_miss(cgraph, i);
                ggml_cuda_op_soft_max(ctx, dst);
            }
            break;
        case GGML_OP_SOFT_CAP_MAX:
            ggml_cuda_op_soft_cap_max(ctx, dst);
            break;
        case GGML_OP_ROPE:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_ROPE &&
                ggml_cuda_op_rope_rope(ctx, dst, cgraph->nodes[i+2])) {
                i += 2;
            }
            else if (fusion && i + 1 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_ROPE &&
                ggml_cuda_op_rope_rope(ctx, dst, cgraph->nodes[i+1])) {
                i += 1;
            } else {
                ggml_cuda_op_rope(ctx, dst);
            }
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROPE_FAST:
            if (fusion && i + 3 < cgraph->n_nodes &&
               (cgraph->nodes[i+1]->op == GGML_OP_RESHAPE || cgraph->nodes[i+1]->op == GGML_OP_VIEW) &&
               (cgraph->nodes[i+2]->op == GGML_OP_RESHAPE || cgraph->nodes[i+2]->op == GGML_OP_VIEW) &&
                cgraph->nodes[i+3]->op == GGML_OP_ROPE_FAST &&
                ggml_cuda_op_fused_rope_fast(ctx, dst, cgraph->nodes[i+3])) {
                i += 3;
            }
            else if (fusion && i + 2 < cgraph->n_nodes &&
               (cgraph->nodes[i+1]->op == GGML_OP_RESHAPE || cgraph->nodes[i+1]->op == GGML_OP_VIEW) &&
                cgraph->nodes[i+2]->op == GGML_OP_ROPE_FAST &&
                ggml_cuda_op_fused_rope_fast(ctx, dst, cgraph->nodes[i+2])) {
                i += 2;
            }
            else if (fusion && i + 1 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_ROPE_FAST   &&
                ggml_cuda_op_fused_rope_fast(ctx, dst, cgraph->nodes[i+1])) {
                i += 1;
            }
            else {
                ggml_cuda_op_rope_fast(ctx, dst);
            }
            break;
        case GGML_OP_ROPE_CACHE:
            ggml_cuda_op_rope_cache(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_SCALE &&
                cgraph->nodes[i+2]->op == GGML_OP_DIV &&
                cgraph->nodes[i+1]->src[0] == dst &&
                cgraph->nodes[i+2]->src[1] == cgraph->nodes[i+1] &&
                cgraph->nodes[i+2]->src[0] == dst->src[0] && ops_are_same_device(cgraph, i, i+2)) {
                ggml_cuda_op_sum_rows_div(ctx, cgraph->nodes[i+2]);
                i += 2;
            }
            else if (fusion && i + 1 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_DIV &&
                cgraph->nodes[i+1]->src[1] == dst &&
                cgraph->nodes[i+1]->src[0] == dst->src[0] && ops_are_same_device(cgraph, i, i+1)) {
                ggml_cuda_op_sum_rows_div(ctx, cgraph->nodes[i+1]);
                ++i;
            } else {
                ggml_cuda_op_sum_rows(ctx, dst);
            }
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            if (fusion && i + 5 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_GET_ROWS &&
                cgraph->nodes[i+3]->op == GGML_OP_RESHAPE &&
                cgraph->nodes[i+4]->op == GGML_OP_SOFT_MAX &&
                cgraph->nodes[i+5]->op == GGML_OP_RESHAPE && ops_are_same_device(cgraph, i, i+4)) {
                cuda_openai_experts(ctx, dst, cgraph->nodes[i+4]);
                i += 5;
            } else {
                ggml_cuda_op_argsort(ctx, dst);
            }
            break;
        case GGML_OP_ARGSORT_THRESH:
            ggml_cuda_op_argsort_thresh(ctx, dst);
            break;
        case GGML_OP_GROUPED_TOPK:
            ggml_cuda_op_grouped_topk(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_DELTA_NET:
            ggml_cuda_op_delta_net(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_SPLIT_SINKHORN:
            ggml_cuda_op_dsv4_hc_split_sinkhorn(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_WEIGHTED_SUM:
            ggml_cuda_op_dsv4_hc_weighted_sum(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_EXPAND:
            ggml_cuda_op_dsv4_hc_expand(ctx, dst);
            break;
        case GGML_OP_MASK_TO_IDX:
            ggml_cuda_op_mask_to_idx(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        default:
            return false;
    }

#if 0
    if (auto err = cudaStreamSynchronize(ctx.stream()); err != cudaSuccess) {
        GGML_CUDA_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }
#endif

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_CUDA_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

#if IK_PRINT_TIMING
    if (auto err = cudaStreamSynchronize(ctx.stream()); err != cudaSuccess) {
        GGML_CUDA_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }
    int64_t tim2 = ggml_time_us();
    printf("%s(%s): %d us\n", ggml_op_name(dst->op), dst->name, (int)(tim2 - tim1));
#endif

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

GGML_CALL static const char * ggml_backend_cuda_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

GGML_CALL static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_cuda_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return ggml_backend_cuda_buffer_type(cuda_ctx->device);
}

GGML_CALL static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    ggml_cuda_set_device(cuda_ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

GGML_CALL static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

GGML_CALL static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(src->buffer) || !ggml_backend_buffer_is_cuda(dst->buffer)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *)backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *)backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *)buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *)buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_CUDA_LOG_WARN("%s: backend and buffer devices do not match\n", __func__);
#endif
        return false;
    }

    if (backend_src != backend_dst) {
        ggml_cuda_pool_alloc<half> tmp_src(cuda_ctx_src->pool());
        ggml_cuda_pool_alloc<half> tmp_dst(cuda_ctx_dst->pool());
        bool needs_f16_f32_copy = false;
        // copy on src stream
        if (cuda_ctx_src->device == cuda_ctx_dst->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            if (false && src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && dst->ne[1] >= 32) {
                //
                // The goal here is to reduce traffic between GPU's, which is entirely non-negligible
                // for prompt processing.
                // We cast the tensor to be copied to f16, copy the f16 data peer-to-peer
                // and then cast back to f32 on the destination side.
                // The cost for converting to/from f16 is much lower than the cost of copying
                // two times more data over PCI-E (well, at least the 30 GB/s PCI-E I have).
                // But for some reason the following is slower.
                // Can somebody tell me why?
                //

                ggml_cuda_set_device(cuda_ctx_dst->device);
                tmp_dst.alloc(ggml_nelements(dst));

                ggml_cuda_set_device(cuda_ctx_src->device);
                tmp_src.alloc(ggml_nelements(src));

                auto src_f16 = *src;
                src_f16.type = GGML_TYPE_F16;
                for (int i = 0; i < 4; ++i) src_f16.nb[i] /= 2;
                src_f16.data = tmp_src.get();

                ggml_cuda_cpy(*cuda_ctx_src, src, &src_f16, true);

                CUDA_CHECK(cudaMemcpyPeerAsync(tmp_dst.ptr, cuda_ctx_dst->device, src_f16.data, cuda_ctx_src->device, ggml_nbytes(&src_f16), cuda_ctx_src->stream()));

                needs_f16_f32_copy = true;

            } else {
#ifdef GGML_USE_NCCL__
                auto & info = ggml_cuda_info();
                auto nbytes = ggml_nbytes(src);
                ncclGroupStart();
                ggml_cuda_set_device(cuda_ctx_src->device);
                auto status1 = ncclSend(src->data, nbytes, ncclUint8, cuda_ctx_dst->device, info.nccl_coms[cuda_ctx_src->device],
                        info.all_ctx[cuda_ctx_src->device]->stream());
                ggml_cuda_set_device(cuda_ctx_dst->device);
                auto status2 = ncclRecv(dst->data, nbytes, ncclUint8, cuda_ctx_src->device, info.nccl_coms[cuda_ctx_dst->device],
                        info.all_ctx[cuda_ctx_dst->device]->stream());
                ncclGroupEnd();
                GGML_ASSERT(status1 == ncclSuccess && status2 == ncclSuccess);
                return true;
#else
                ggml_cuda_set_device(cuda_ctx_src->device);
                CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, cuda_ctx_dst->device, src->data, cuda_ctx_src->device, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif
            }
#endif
        }

        // record event on src stream after the copy
        ggml_cuda_set_device(cuda_ctx_src->device);
        if (!cuda_ctx_src->copy_event) {
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        ggml_cuda_set_device(cuda_ctx_dst->device);
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
        if (needs_f16_f32_copy) {
            auto dst_f16 = *dst;
            dst_f16.type = GGML_TYPE_F16;
            for (int i = 0; i < 4; ++i) dst_f16.nb[i] /= 2;
            dst_f16.data = tmp_dst.get();
            ggml_cuda_cpy(*cuda_ctx_dst, &dst_f16, dst, true);
        }
    } else {
        // src and dst are on the same backend
        // printf("Why is this being invoked?\n");
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

GGML_CALL static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);
    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

#ifdef USE_CUDA_GRAPH

// PXA_CUDA_GRAPH_BATCH (S2, 2026-07-07): opt-in CUDA-graph capture for small multi-token
// decode batches (spec-decode verify shapes, np>1 coalesced multi-slot decode). Default OFF; the
// n_tokens==1 behavior is byte-identical when the env is unset. See FORK-OPTIMIZATION-SURVEY F1/F2.
static bool pxa_cuda_graph_batch_enabled() {
    static const bool on = [] {
        const char * e = getenv("PXA_CUDA_GRAPH_BATCH");
        if (!e || atoi(e) == 0) return false;
        // The E2 union probe (PXA_MOE_DEBUG) does a cudaMemcpy D2H + stream sync INSIDE the MoE op,
        // and the A1 shadow-verify (PXA_MOE_GROUPED_VERIFY) relies on device-printf flushing at
        // stream syncs -- both are incompatible with (or defeated by) graph capture/replay.
        // (PXA_MOE_GROUPED_VERIFY goes through its value-tested resolver since 2026-07-30 —
        // a presence test here would disable graphs on an explicit =0.)
        if (getenv("PXA_MOE_DEBUG") || pxa_moe_grouped_shadow()) return false;
        return true;
    }();
    return on;
}
static int pxa_cuda_graph_batch_max_ny() {
    static const int v = [] {
        int m = 8; // the fast-TG dispatch ceiling in ggml_cuda_moe_up_gate_unary
        if (const char * e = getenv("PXA_MOE_FASTTG_MAX_NY"))       { int t = atoi(e); if (t < m) m = t; }
        if (const char * e = getenv("PXA_CUDA_GRAPH_BATCH_MAX_NY")) { int t = atoi(e); if (t >= 1 && t < m) m = t; }
        return m;
    }();
    return v;
}

static bool pxa_cuda_graph_log_enabled() {
    static const bool on = getenv("PXA_CUDA_GRAPH_LOG") != nullptr;
    return on;
}
static int pxa_cuda_graph_log_level() {
    static const int v = [] { const char * e = getenv("PXA_CUDA_GRAPH_LOG"); return e ? atoi(e) : 0; }();
    return v;
}

// PXA_CUDA_GRAPH_V2 (G1, 2026-07-19): fixed graph-cache semantics, opt-in, default OFF.
// =0/unset is byte-identical to the pre-G1 behavior. When on:
//   1. the shape-FNV cache key is ALWAYS used (not only under PXA_CUDA_GRAPH_BATCH) and folds the
//      (op, ne1, ne2) of EVERY node -- each shape class (prefill chunk, decode variant, MTP
//      warmup/verify, np2 ny=2) owns a stable slot instead of all colliding on nodes[0];
//   2. disable_due_to_too_many_updates is replaced by a cooldown (PXA_CUDA_GRAPH_REARM evals,
//      default 256; 0 = keep the permanent disable) so early churn can't kill replay forever;
//   3. the capturability check runs BEFORE the property store, so an uncapturable graph (prefill
//      MoE ny>8) no longer clobbers the stored decode properties at a shared/collided key;
//   4. the per-context graph cache is LRU-capped (PXA_CUDA_GRAPH_LRU slots, default 8).
static bool pxa_cuda_graph_v2_enabled() {
    // NOTE (2026-07-21): whole-token graph replay is measured NEGATIVE on BOTH box arches with
    // captures verified firing (PXA_CUDA_GRAPH_LOG): V100 −2..−4%, P100 −3.9% (65.0→62.5 t/s,
    // public PXQ2, replays=396/400 tokens, byte-identical). An earlier "+3.5% P100" reading was
    // NOISE from a config where the cc<CC_AMPERE arch gate silently kept captures at 0 — always
    // verify captures>0 before believing a graph number. Decode is GPU-busy; replay bookkeeping
    // is pure tax on these cards. Stays env-only opt-in for instrumentation/diagnostics.
    //
    // RE-CONFIRMED (2026-08-28) on a very different graph: qwen4exp across six cards, 5817 nodes
    // and ~140k sub-15us elementwise kernels per run — i.e. exactly the shape that looks like it
    // should be launch-bound. It is not. Same 127-token measurement either side, captures
    // verified firing (replays=882): 26.43 -> 25.68 t/s, −2.8%. The per-op profile explains why:
    // MUL_MAT is 52.4% and MOE_FUSED_UP_GATE 16.7% of GPU time, so the tiny ops are interleaved
    // with real work rather than sitting in launch gaps. Do not re-test this without a profile
    // showing idle gaps first. Beware short runs: two arms of that sweep stopped at 53-57 tokens
    // and their t/s is NOT comparable to a 127-token baseline.
    static const bool on = [] { const char * e = getenv("PXA_CUDA_GRAPH_V2"); return e && atoi(e) != 0; }();
    return on;
}
// PXA_CUDA_GRAPH_PREFILL (2026-09-02): CUDA-graph capture/replay for MULTI-TOKEN PREFILL ubatches
// on sm_60/sm_70, ported from mainline llama.cpp 9400c8946e4da5e7694f2c26d6d4e50e14b690fa (MIT,
// ggml/src/ggml-cuda/ggml-cuda.cu: ggml_cuda_graph_set_enabled / ggml_backend_cuda_graph_compute).
//
// Three deltas vs. our pre-existing (ik-lineage) graph path, which refused to capture at all below
// Ampere:
//   1. the arch gate. Ours was cc < CC_AMPERE; mainline's is cc < CC_VOLTA. Neither is a hardware
//      limit -- stream capture works on sm_60 -- both are policy, recorded from DECODE measurements.
//      Under this flag the gate is lifted for cc >= CC_PASCAL and the policy question is settled by
//      the prefill measurement instead.
//   2. capture admission. Ours captured on the FIRST eval of a key and permanently disabled the key
//      after 4 consecutive property changes; a prefill whose n_kv grows 256 cells per ubatch trips
//      that immediately, so it paid N captures and then banned itself. Mainline's rule (ported to
//      ggml_cuda_graph::warmup_complete) is the inverse: never capture until two consecutive evals
//      have IDENTICAL node properties, and on any later change drop back to eager rather than
//      recapture. Churn then costs nothing beyond the property compare.
//   3. the cache key. Prefill and decode ubatches must not share a slot, so the V2 shape-FNV key is
//      forced on whenever this flag is set.
// Per-ubatch parameters are refreshed exactly as mainline does it: node properties (dst/src data
// pointers, ne, nb, op_params) are re-read every eval, and a change forces re-capture followed by
// cudaGraphExecUpdate (update_cuda_graph_executable) rather than any hand-patching of kernel args.
// The KV-cache destination pointers keep our existing cpy-indirection path, which is strictly more
// permissive than mainline's (mainline dropped indirection and re-captures instead).
// MEASURED VERDICT (2026-09-02, Qwable-27B-PXQ4core, -b/-ub 2048, -fa on, n=3, PXQ4):
//   V100 pair  20801-tok prefill:  581.6 / 576.5 t/s (flag off, two identical arms) vs 588.9 (on).
//   P100 pair  20801-tok prefill:  202.2 (off) vs 203.4 (arch gate lifted, old admission policy)
//                                          vs 202.5 (shape-keyed V2 policy).
// Every delta sits at or under the run-to-run spread (the two identical V100 off-arms differ by
// 0.9% at 20801 and 5.4% at 3121), so this wins on NEITHER arch and stays default OFF.
// The mechanism does work: the admission rule cuts wasted captures from 170 to 31 over the same
// run while keeping the same 7 replays. 7 replays is the whole story -- a prefill ubatch's
// attention nodes carry ne[0] = n_kv, which grows 256 cells per ubatch, so two consecutive
// prefill evals are never property-identical and the captured graph is essentially never reached.
// The replays that do occur come from re-prefilling an IDENTICAL prompt (bench reps), not from
// anything a server sees in production. Do not re-test without first showing prefill idle gaps in
// a profile, and do not believe a graph number without checking captures/replays in PXA_CUDA_GRAPH_LOG.
static bool pxa_cuda_graph_prefill_enabled() {
    static const bool on = [] { const char * e = getenv("PXA_CUDA_GRAPH_PREFILL"); return e && atoi(e) != 0; }();
    return on;
}

// PXA_CUDA_GRAPH_DECODE (2026-09-03): the SAME mainline admission rule, applied to the M=1 decode
// graph on Volta. Opt-in, DEFAULT OFF -- see the measured verdict at the bottom of this block.
// The mechanism works and the admission rule does what it should; the path underneath does not.
//
// Why this is not the whole-token graph replay that measured -2..-4% (see the note on
// pxa_cuda_graph_v2_enabled): that path captured on the first eval of a key and re-captured on
// every property change. On a decode step the KV-dependent node extents move, so it paid a
// capture, a cudaGraphExecUpdate, or a full re-instantiate on a large fraction of tokens and the
// bookkeeping ate the launch saving. Mainline's rule inverts it: never capture until two
// CONSECUTIVE evals have byte-identical node properties, and on any later change fall back to
// eager and re-arm rather than re-capture.
//
// The reason that rule pays here and did not pay for prefill is the KV padding. With -fa on,
// llama_kv_cache_get_padding() is 256, so kv_self.n is GGML_PAD(max_cell, 256): across a 256-token
// generation the attention nodes' n_kv is CONSTANT and only steps at a 256-cell boundary. Two
// consecutive decode evals are therefore property-identical almost always -- the opposite of a
// prefill ubatch, whose n_kv grows 256 cells EVERY ubatch so that two consecutive evals are never
// identical and the captured graph is essentially never reached (measured: 170 captures, 7
// replays). Decode captures once per 256-token band and replays the rest of the band; the mask and
// positions are graph INPUTS written into fixed device buffers, and the KV write destination goes
// through our existing cpy-indirection, so neither forces a re-capture.
//
// Correctness: replay is not an approximation. is_cuda_graph_update_required() compares every
// node's op / ne / nb / op_params / data pointers against the stored properties BEFORE any replay,
// so a stale graph can never run at the wrong shape; a mismatch reverts to eager. The greedy sha
// gate is run with this on.
//
// MEASURED VERDICT (2026-09-03, Qwable-27B-PXQ4core, 2x V100 -sm layer, -fa on, -b/-ub 2048).
// DEFAULT OFF on both counts -- it is neither correct nor faster on this path:
//
//  1. NOT FASTER. Decode at fill 8, same session, same build, n=5 per arm:
//       graphs off  38.38 / 38.13 t/s      graphs on  37.24 / 36.63 / 36.01 / 37.86 t/s
//     Capture and replay are verifiably live (PXA_CGRAPH SUMMARY captures=87 replays=4569,
//     eager{disabled=0 uncapturable=0}), so this is a real measurement of the mechanism, not a
//     phantom lever. The reason it does not pay is visible in PXA_HOST_TIMING: the 22.1 ms of
//     "submit" per 28.3 ms token is NOT mostly kernel-launch cost. With -sm layer the scheduler
//     runs several splits per device per token and the host work BETWEEN splits -- the blocking
//     H2D input copies in ggml_backend_sched_copy_inputs and the per-split event bookkeeping --
//     is outside any captured graph. Replay removes the launches and leaves the actual cost.
//
//  2. NOT CORRECT on the server lifecycle. The SECOND request into a warm slot aborts with
//     "CUDA error: an illegal memory access was encountered" in ggml_backend_cuda_synchronize.
//     Repro: boot the server, POST /completion twice (n_predict 48 is enough); the second one
//     dies right after the "kv cache rm [p0, end)" and the pxa_reserve_real_graph re-reserve that
//     follows it, i.e. a captured exec is replayed across a compute-buffer re-reserve that the
//     node-property compare does not catch. This is the fork's pre-existing graph path, NOT the
//     delta-net fusions: it reproduces identically at PXA_FUSE_DELTANET=3 (bits 2|3 off) and
//     disappears entirely at PXA_CUDA_GRAPH_DECODE=0 with PXA_FUSE_DELTANET=15. Anyone who
//     re-enables this must fix that re-reserve first; the arch gate had been hiding it.
static bool pxa_cuda_graph_decode_enabled() {
    static const bool on = [] {
        const char * e = getenv("PXA_CUDA_GRAPH_DECODE");
        return e && atoi(e) != 0;
    }();
    return on;
}

// True when the mainline capture-admission policy (warmup_complete / revert-to-eager) is in force,
// from either flag. Both also force the V2 shape-FNV cache key on, so a prefill ubatch and a decode
// step can never collide on one cache slot.
static bool pxa_cuda_graph_mainline_policy() {
    return pxa_cuda_graph_prefill_enabled() || pxa_cuda_graph_decode_enabled();
}

static int pxa_cuda_graph_rearm_evals() {
    static const int v = [] { const char * e = getenv("PXA_CUDA_GRAPH_REARM"); return e ? atoi(e) : 256; }();
    return v;
}
static size_t pxa_cuda_graph_lru_cap() {
    static const size_t v = [] { const char * e = getenv("PXA_CUDA_GRAPH_LRU");
        long t = e ? atol(e) : 8; return (size_t)(t < 2 ? 2 : t); }();
    return v;
}

// P0 instrumentation: honest global counters + atexit summary (active only under PXA_CUDA_GRAPH_LOG).
struct pxa_cgraph_stats_t {
    std::atomic<long> captures{0};
    std::atomic<long> replays{0};
    std::atomic<long> eager_disabled{0};   // one of the disable flags / arch gate
    std::atomic<long> eager_cooldown{0};   // v2 cooldown active
    std::atomic<long> eager_uncapturable{0}; // compat check said no
    std::atomic<long> disables_arch{0};
    std::atomic<long> disables_too_many{0};
    std::atomic<long> disables_capture_fail{0};
    std::atomic<long> cooldowns_armed{0};
    std::atomic<long> lru_evictions{0};
};
static pxa_cgraph_stats_t & pxa_cgraph_stats() {
    static pxa_cgraph_stats_t s;
    return s;
}
static void pxa_cgraph_atexit_summary() {
    auto & s = pxa_cgraph_stats();
    fprintf(stderr, "PXA_CGRAPH SUMMARY captures=%ld replays=%ld eager{disabled=%ld cooldown=%ld uncapturable=%ld} "
            "disables{arch=%ld too_many=%ld capture_fail=%ld} cooldowns_armed=%ld lru_evictions=%ld\n",
            s.captures.load(), s.replays.load(), s.eager_disabled.load(), s.eager_cooldown.load(),
            s.eager_uncapturable.load(), s.disables_arch.load(), s.disables_too_many.load(),
            s.disables_capture_fail.load(), s.cooldowns_armed.load(), s.lru_evictions.load());
    fflush(stderr);
}
static void pxa_cgraph_stats_arm_atexit() {
    static std::once_flag once;
    std::call_once(once, [] { atexit(pxa_cgraph_atexit_summary); });
}

static inline const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    if (!pxa_cuda_graph_batch_enabled() && !pxa_cuda_graph_v2_enabled() && !pxa_cuda_graph_mainline_policy()) {
        return cgraph->nodes[0];
    }
    // PXA_CUDA_GRAPH_BATCH shape-keyed cache: with multi-token capture on, DIFFERENT batch shapes
    // alternate step-to-step (verify depth K varies with draft acceptance; server slots join/leave).
    // The host cgraph arena is rebuilt at the same base address after every scheduler reset, so
    // keying on nodes[0] alone maps ALL shapes onto ONE entry -> re-capture on every shape flip ->
    // "too many consecutive updates" permanently disables graphs for the shared key (losing even the
    // banked single-token win). Fold a cheap shape/topology signature into the key so each distinct
    // shape owns a stable entry and replays its own captured graph.
    // Correctness does NOT depend on this key: is_cuda_graph_update_required() compares every node's
    // op/ne/nb/addresses before any replay, so a key collision can only cost a re-capture -- a stale
    // graph can never be replayed at the wrong shape.
    uint64_t h = 0xcbf29ce484222325ULL; // FNV-1a
    auto fold = [&h](uint64_t x) { h ^= x; h *= 0x100000001b3ULL; };
    fold((uint64_t)(uintptr_t)cgraph->nodes[0]);
    fold((uint64_t)cgraph->n_nodes);
    const int n = cgraph->n_nodes;
    // V2: fold (op, ne1, ne2) over ALL nodes (~us-scale for 2500 nodes) so distinct topologies that
    // share a 16-node prefix (e.g. the 2498 vs 2468 decode variants) get distinct keys. Legacy
    // BATCH-only mode keeps the 16-node prefix fold (byte-identical to the shipped behavior).
    const int fold_n = (pxa_cuda_graph_v2_enabled() || pxa_cuda_graph_mainline_policy()) ? n : (n < 16 ? n : 16);
    for (int i = 0; i < fold_n; ++i) {
        const ggml_tensor * t = cgraph->nodes[i];
        fold((uint64_t)t->op); fold((uint64_t)t->ne[1]); fold((uint64_t)t->ne[2]);
    }
    if (n > 0) {
        const ggml_tensor * t = cgraph->nodes[n-1];
        fold((uint64_t)t->op); fold((uint64_t)t->ne[0]); fold((uint64_t)t->ne[1]); fold((uint64_t)t->ne[2]);
    }
    return (const void *)(uintptr_t)h;
}

static inline ggml_cuda_graph * ggml_cuda_get_graph(ggml_backend_cuda_context & ctx, const void * key) {
    auto & graph = ctx.cuda_graphs[key];
    if (!graph) {
        graph = std::make_unique<ggml_cuda_graph>();
        // PXA_CUDA_GRAPH_V2 P1.4: bound the keyed cache. Each captured 2,500-node exec holds device
        // memory; evict the least-recently-used entry (never the one we just created) beyond the cap.
        if ((pxa_cuda_graph_v2_enabled() || pxa_cuda_graph_mainline_policy()) && ctx.cuda_graphs.size() > pxa_cuda_graph_lru_cap()) {
            const void * lru_key = nullptr;
            uint64_t lru_use = UINT64_MAX;
            for (auto & kv : ctx.cuda_graphs) {
                if (kv.first == key || !kv.second) continue;
                if (kv.second->last_use < lru_use) { lru_use = kv.second->last_use; lru_key = kv.first; }
            }
            if (lru_key) {
                if (pxa_cuda_graph_log_enabled()) {
                    fprintf(stderr, "PXA_CGRAPH lru-evict dev=%d key=%p (cache=%zu cap=%zu)\n",
                            ctx.device, lru_key, ctx.cuda_graphs.size(), pxa_cuda_graph_lru_cap());
                }
                pxa_cgraph_stats().lru_evictions++;
                ctx.cuda_graphs.erase(lru_key);
            }
        }
    }
    auto * g = ctx.cuda_graphs[key].get();
    g->last_use = ++ctx.cuda_graph_eval_no;
    return g;
}

static bool check_node_graph_compatibility_and_refresh_copy_ops(ggml_backend_cuda_context * cuda_ctx,
    ggml_cuda_graph * graph, ggml_cgraph * cgraph, bool use_cuda_graph, cudaStream_t stream) {

    // Loop over nodes in GGML graph to obtain info needed for CUDA graph
    graph->cpy_dest_ptrs.clear();

    const std::string gemma3n_per_layer_proj_src0_name = "inp_per_layer_selected";
    const std::string gemma3n_per_layer_proj_src1_name = "per_layer_proj";
    const std::string ffn_moe_gate_bias_prefix = "ffn_moe_gate_biased";
    const std::string ffn_moe_up_bias_prefix = "ffn_moe_up_biased";
    const std::string ffn_moe_down_bias_prefix = "ffn_moe_down_biased";

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_is_noop(node)) continue;

        if (node->op == GGML_OP_REDUCE) {
            // PXA_REDUCE_CAPTURE: allow capturing the cross-device ring/direct reduce once
            // every participating device's copy_event AND reduce_kickoff_event already exist
            // (the first reduce of a session runs eager and creates them; capture forbids alloc).
            static const bool _pxa_rc = getenv("PXA_REDUCE_CAPTURE") != nullptr;
            if (!_pxa_rc) { use_cuda_graph = false; break; }
            auto & _ri = ggml_cuda_info();
            int _rn = node->op_params[1];
            bool _ready = true;
            for (int _rj = 0; _rj < _rn && _ready; ++_rj) {
                if (node->src[_rj]) {
                    auto _c = _ri.all_ctx[_rj];
                    if (!_c || !_c->copy_event || !_c->reduce_kickoff_event) _ready = false;
                }
            }
            if (!_ready) { use_cuda_graph = false; break; }
            continue; // reduce is capturable this token
        }

        if (node->op == GGML_OP_MUL_MAT_ID) {
            // PXA_CUDA_GRAPH_MOE: the original guard disabled capture whenever n_expert_used>1
            // (src[2]->ne[0]!=1) -> fired on EVERY MoE token (qwen35moe routes 8 experts) so graph
            // capture NEVER engaged. But the expert ids are read ON-DEVICE by the Fast-TG kernel
            // (capturable); only the host-grouping path (large batch / non-quant) does a real
            // cudaStreamSynchronize and is truly uncapturable. Relax (opt-in PXA_CUDA_GRAPH_MOE) to
            // the predicate the Fast-TG dispatch itself uses, so MoE decode can be captured.
            const bool pxa_uncapturable = node->ne[2] > 8 || node->src[1]->ne[2] != 1 ||
                                          !ggml_is_quantized(node->src[0]->type);
            const bool mmid_disable = getenv("PXA_CUDA_GRAPH_MOE")
                ? pxa_uncapturable
                : (node->ne[2] != 1 || node->src[2]->ne[0] != 1);
            if (mmid_disable) {
                use_cuda_graph = false; // This node type is not supported by CUDA graph capture
#ifndef NDEBUG
                GGML_CUDA_LOG_DEBUG("%s(%s): disabling CUDA graphs due to unsupported node type %ld %ld\n",
                        __func__, node->src[0]->name, node->ne[2], node->src[2]->ne[0]);
#endif
            }
        }
        if (node->op == GGML_OP_MOE_FUSED_UP_GATE) {
            auto src0_1 = node->src[0];
            auto src0_2 = node->src[1];
            auto src1   = node->src[2];
            // PXA_CUDA_GRAPH_BATCH (S2): the original gate restricted capture to single-token MoE
            // (src1->ne[2]==1), so every multi-token decode step (MTP/ngram verify batches, np>1
            // coalesced multi-slot decode) ran permanently eager and forfeited the banked graph win
            // (survey F1). The fast-TG iy-loop and the A1 grouped up+gate path are both
            // host-sync-free for ne[2] in 2..8 (pool allocs are legal under relaxed capture and the
            // VMM pool is pointer-stable), so those shapes are capturable. The allowed Ny range
            // follows the execution dispatch (PXA_MOE_FASTTG_MAX_NY), and the same-device buffer
            // predicate is mirrored from ggml_cuda_moe_up_gate_unary, so we never capture a shape
            // that would fall through to the host-syncing general/mmq-id path.
            bool moe_capturable =
                src1->ne[1] == 1 && src1->ne[3] == 1 && src1->type == GGML_TYPE_F32 &&
                ggml_is_quantized(src0_1->type) && (!src0_2 || ggml_is_quantized(src0_2->type));
            if (moe_capturable && src1->ne[2] != 1) {
                moe_capturable = pxa_cuda_graph_batch_enabled() &&
                    src1->ne[2] >= 2 && src1->ne[2] <= pxa_cuda_graph_batch_max_ny() &&
                    src0_1->buffer && src1->buffer && node->buffer && (!src0_2 || src0_2->buffer) &&
                    ggml_backend_buffer_is_cuda(src0_1->buffer) &&
                    (!src0_2 || ggml_backend_buffer_is_cuda(src0_2->buffer)) &&
                    ggml_backend_buffer_is_cuda(src1->buffer) &&
                    ggml_backend_buffer_is_cuda(node->buffer) &&
                    ((ggml_backend_cuda_buffer_context *) src0_1->buffer->context)->device == cuda_ctx->device &&
                    (!src0_2 || ((ggml_backend_cuda_buffer_context *) src0_2->buffer->context)->device == cuda_ctx->device) &&
                    ((ggml_backend_cuda_buffer_context *) src1->buffer->context)->device == cuda_ctx->device &&
                    ((ggml_backend_cuda_buffer_context *) node->buffer->context)->device == cuda_ctx->device;
            }
            // PXQ4: only the fused decode path is device-only/capturable. If the type is PXQ4
            // but that path would decline (env off, wrong unary, unaligned dims, smem), decode
            // falls to the host-syncing per-expert loop -> must NOT capture.
            if (moe_capturable && (src0_1->type == GGML_TYPE_PXQ4 || src0_1->type == GGML_TYPE_PXQ4HQ ||
                                   src0_1->type == GGML_TYPE_PXQ2 || src0_1->type == GGML_TYPE_PXQ3 ||
                                   src0_1->type == GGML_TYPE_PXQ1 ||
                                   src0_1->type == GGML_TYPE_PXQ6)) {
                const ggml_unary_op puop = (ggml_unary_op)node->op_params[0];
                const int cfu = pxa_pxq_fmt(src0_1->type);
                const int cfg = src0_2 ? pxa_pxq_fmt(src0_2->type) : PXA_PXQ_FMT_NONE;
                const bool pair_ok = cfu != PXA_PXQ_FMT_NONE && cfg != PXA_PXQ_FMT_NONE &&
                    (cfu == cfg || ((cfu == PXA_PXQ_FMT_P1 || cfu == PXA_PXQ_FMT_P2 ||
                                     cfu == PXA_PXQ_FMT_P3 || cfu == PXA_PXQ_FMT_P6) &&
                                    (cfg == PXA_PXQ_FMT_P1 || cfg == PXA_PXQ_FMT_P2 ||
                                     cfg == PXA_PXQ_FMT_P3 || cfg == PXA_PXQ_FMT_P6)));
                moe_capturable = pair_ok &&
                    src0_2 &&
                    (puop == GGML_UNARY_OP_SWIGLU_OAI || puop == GGML_UNARY_OP_SILU) &&
                    src0_1->ne[1] % PXQ4_BM == 0 && src0_1->ne[0] % PXQ4_QK == 0 &&
                    (size_t)src0_1->ne[0]*sizeof(float) + 2*PXQ4_MMV_KSEG*64*sizeof(float) <= 46*1024;
            }
            if (!moe_capturable) {
                use_cuda_graph = false;
            } else {
                if (i < cgraph->n_nodes-1) {
                    auto next = cgraph->nodes[i+1];
                    if (next->op == GGML_OP_MUL_MAT_ID && ggml_is_quantized(next->src[0]->type)) {
                        ++i;
                    }
                }
            }
        }

        // Why was this needed? Leaving it in place but disabled in case it is actually needed.
        if (false && node->op == GGML_OP_ADD &&
            node->src[1] && node->src[1]->ne[1] > 1 &&
            (node->src[0] ? node->src[0]->name != gemma3n_per_layer_proj_src0_name : true) &&
            (node->src[1] ? node->src[1]->name != gemma3n_per_layer_proj_src1_name : true) &&
            strncmp(node->name, ffn_moe_gate_bias_prefix.c_str(), ffn_moe_gate_bias_prefix.size()) != 0 &&
            strncmp(node->name, ffn_moe_up_bias_prefix.c_str(), ffn_moe_up_bias_prefix.size()) != 0 &&
            strncmp(node->name, ffn_moe_down_bias_prefix.c_str(), ffn_moe_down_bias_prefix.size()) != 0) {
            // disable CUDA graphs for batch size > 1 for now while excluding the matrix-matrix addition as part of Gemma3n's `project_per_layer_input` operation
            // by means of matching node names. See
            // https://github.com/ggml-org/llama.cpp/blob/f9a31eea06a859e34cecb88b4d020c7f03d86cc4/src/llama-model.cpp#L10199-L10241 and
            // https://github.com/huggingface/transformers/blob/bda75b4011239d065de84aa3e744b67ebfa7b245/src/transformers/models/gemma3n/modeling_gemma3n.py#L1773,
            // Generally, changes in batch size or context size can cause changes to the grid size of some kernels.
            use_cuda_graph = false;
#ifndef NDEBUG
            GGML_CUDA_LOG_DEBUG("%s: disabling CUDA graphs due to batch size > 1 [%s] [%ld %ld %ld %ld]\n", __func__, node->name, node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
#endif
        }

        if (node->op == GGML_OP_CPY) {

            // Store the pointers which are updated for each token, such that these can be sent
            // to the device and accessed using indirection from CUDA graph
            graph->cpy_dest_ptrs.push_back((char *) node->src[1]->data);

            // store a pointer to each copy op CUDA kernel to identify it later
            void * ptr = ggml_cuda_cpy_fn(node->src[0], node->src[1]);
            if (!ptr) {
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_CUDA_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported copy op\n", __func__);
#endif
            }
        }
        if (!use_cuda_graph) {
            break;
        }
    }

    if (use_cuda_graph) {
        graph->use_cpy_indirection = true;
        // copy pointers to GPU so they can be accessed via indirection within CUDA graph
        ggml_cuda_cpy_dest_ptrs_copy(graph, graph->cpy_dest_ptrs.data(), graph->cpy_dest_ptrs.size(), stream);
    }

    return use_cuda_graph;
}

static void set_ggml_graph_node_properties(ggml_tensor * node, ggml_graph_node_properties * graph_node_properties) {
    graph_node_properties->node_address = node->data;
    graph_node_properties->node_op = node->op;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        graph_node_properties->ne[i] = node->ne[i];
        graph_node_properties->nb[i] = node->nb[i];
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        graph_node_properties->src_address[i] = node->src[i] ? node->src[i]->data : nullptr;
    }
    memcpy(graph_node_properties->op_params, node->op_params, GGML_MAX_OP_PARAMS);
}

static bool ggml_graph_node_has_matching_properties(ggml_tensor * node, ggml_graph_node_properties * graph_node_properties) {
    if (node->data != graph_node_properties->node_address &&
          node->op != GGML_OP_CPY &&
          node->op != GGML_OP_VIEW) {
        return false;
    }

    if (node->op != graph_node_properties->node_op) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (node->ne[i] != graph_node_properties->ne[i]) {
            return false;
        }
        if (node->nb[i] != graph_node_properties->nb[i]) {
            return false;
        }
    }

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node->src[i] &&
            node->src[i]->data != graph_node_properties->src_address[i] &&
            node->op != GGML_OP_VIEW &&
            // PXQ port of upstream ik_llama.cpp PR #2136 (merged 2026-07-17): only a CPY node's
            // DESTINATION (src[1], the KV-cache view whose address legitimately moves between
            // captures) is exempt from the address check. The READ source (src[0]) must force a
            // re-capture when it moves, or the captured graph keeps reading the old address
            // (stale-read bug, e.g. SWA mask/KV re-pointing).
            !(node->op == GGML_OP_CPY && i == 1)
        ) {
            return false;
        }
    }

    if (node->op == GGML_OP_SCALE &&
        memcmp(graph_node_properties->op_params, node->op_params, GGML_MAX_OP_PARAMS) != 0) {
        return false;
    }

    return true;
}

// P0 instrumentation: name WHICH field of WHICH node forced the update (first mismatch only).
static void pxa_cgraph_log_first_mismatch(const void * key, int i, ggml_tensor * node, ggml_graph_node_properties * p) {
    const char * field = "?";
    char detail[192] = {0};
    if (node->op != p->node_op) {
        field = "op";
        snprintf(detail, sizeof(detail), "%s->%s", ggml_op_name(p->node_op), ggml_op_name(node->op));
    } else if (node->data != p->node_address && node->op != GGML_OP_CPY && node->op != GGML_OP_VIEW) {
        field = "data";
        snprintf(detail, sizeof(detail), "%p->%p", p->node_address, node->data);
    } else {
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            if (node->ne[d] != p->ne[d]) {
                field = "ne";
                snprintf(detail, sizeof(detail), "dim%d %lld->%lld", d, (long long)p->ne[d], (long long)node->ne[d]);
                break;
            }
            if (node->nb[d] != p->nb[d]) {
                field = "nb";
                snprintf(detail, sizeof(detail), "dim%d %zu->%zu", d, p->nb[d], node->nb[d]);
                break;
            }
        }
        if (detail[0] == 0) {
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                if (node->src[s] && node->src[s]->data != p->src_address[s] &&
                    node->op != GGML_OP_VIEW &&
                    !(node->op == GGML_OP_CPY && s == 1)) { // PR #2136: keep logger in lockstep with the matcher
                    field = "src";
                    snprintf(detail, sizeof(detail), "src%d(%s) %p->%p", s, node->src[s]->name,
                             p->src_address[s], node->src[s]->data);
                    break;
                }
            }
        }
        if (detail[0] == 0 && node->op == GGML_OP_SCALE &&
            memcmp(p->op_params, node->op_params, GGML_MAX_OP_PARAMS) != 0) {
            field = "op_params";
        }
    }
    fprintf(stderr, "PXA_CGRAPH mismatch key=%p node[%d] name=%s op=%s field=%s %s\n",
            key, i, node->name, ggml_op_name(node->op), field, detail);
}

// props_changed_out (PXA_CUDA_GRAPH_PREFILL): reports ONLY whether the stored node properties
// differ from this evaluation's, with the "we have no executable yet" reason excluded. Mainline's
// warmup rule keys off exactly that distinction -- a first eval has no instance but its properties
// are not yet known to be unstable, and treating the two as one signal makes the rule unsatisfiable.
static bool is_cuda_graph_update_required(ggml_cuda_graph * graph, ggml_cgraph * cgraph, const void * key = nullptr,
                                          bool * props_changed_out = nullptr) {

    bool cuda_graph_update_required = false;
    bool props_changed = false;
    const bool log_on = pxa_cuda_graph_log_enabled();

    if (graph->instance == nullptr) {
        cuda_graph_update_required = true;
        if (log_on) fprintf(stderr, "PXA_CGRAPH update-required key=%p reason=no-instance\n", key);
    }

    // Check if the graph size has changed
    if (graph->ggml_graph_properties.size() != (size_t)cgraph->n_nodes) {
        if (log_on && !cuda_graph_update_required) {
            fprintf(stderr, "PXA_CGRAPH update-required key=%p reason=n_nodes %zu->%d\n",
                    key, graph->ggml_graph_properties.size(), cgraph->n_nodes);
        }
        cuda_graph_update_required = true;
        props_changed = true;
        graph->ggml_graph_properties.resize(cgraph->n_nodes);
    }

    // Loop over nodes in GGML graph to determine if CUDA graph update is required
    // and store properties to allow this comparison for the next token
    // When props_changed_out is requested the property compare must run for EVERY node even if an
    // update is already known to be required (no instance), otherwise the stable/unstable answer is
    // never produced. Without it, the short-circuit below is kept byte-for-byte.
    const bool want_props = props_changed_out != nullptr;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        bool has_matching_properties = true;
        if (!cuda_graph_update_required || (want_props && !props_changed)) {
            has_matching_properties = ggml_graph_node_has_matching_properties(cgraph->nodes[i], &graph->ggml_graph_properties[i]);
        }
        if (!has_matching_properties) {
            if (log_on) pxa_cgraph_log_first_mismatch(key, i, cgraph->nodes[i], &graph->ggml_graph_properties[i]);
            cuda_graph_update_required = true;
            props_changed = true;
        }
        set_ggml_graph_node_properties(cgraph->nodes[i], &graph->ggml_graph_properties[i]);
    }

    if (props_changed_out) {
        *props_changed_out = props_changed;
    }

    return cuda_graph_update_required;
}

static void update_cuda_graph_executable(ggml_cuda_graph * graph) {

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    // PXA_CUDA_GRAPH_V2: treat ANYTHING but clean success as "cannot update in place" and
    // destroy+reinstantiate (CUDA 12.x sm70 partial-update edge cases); legacy behavior asserts
    // on unexpected errors.
    if (stat == cudaErrorGraphExecUpdateFailure || (pxa_cuda_graph_v2_enabled() && stat != cudaSuccess)) {
#ifndef NDEBUG
        GGML_CUDA_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif
        if (pxa_cuda_graph_log_enabled()) {
            fprintf(stderr, "PXA_CGRAPH exec-update-failed (%s) -> reinstantiate\n", cudaGetErrorString(stat));
        }

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif

static void evaluate_and_capture_cuda_graph(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph,
    bool & graph_evaluated_or_captured, bool & use_cuda_graph, bool & cuda_graph_update_required) {
    // flag used to determine whether it is an integrated_gpu
    // TODO
    [[maybe_unused]] const bool integrated = false; //ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ++pxa_g2_eval_serial;   // G2-F3: q8 sidecars are only valid within one graph eval

#ifdef USE_CUDA_GRAPH
    auto graph = use_cuda_graph ? ggml_cuda_get_graph(*cuda_ctx, ggml_cuda_graph_get_key(cgraph)) : nullptr;
#endif

#if IK_PRINT_TIMING
    printf("======================== %s: graph with %d nodes on device %d. time = %ld\n", __func__, cgraph->n_nodes, cuda_ctx->device, ggml_time_us());
#endif
    // PXA_WAVE4_DIAG (PXA_OP_SYNC_CHECK=1): report a sticky CUDA error set BEFORE this graph eval
    // (means the fault came from an external/API path between graphs, not a graph op).
    if (g_pxa_op_check_on < 0) g_pxa_op_check_on = getenv("PXA_OP_SYNC_CHECK") ? 1 : 0;
    if (g_pxa_op_check_on > 0) {
        cudaError_t _perr = cudaGetLastError();
        if (_perr != cudaSuccess) {
            fprintf(stderr, "PXA_WAVE4_DIAG ENTRY-DIRTY dev=%d err=%s (fault set before graph eval)\n",
                    cuda_ctx->device, cudaGetErrorString(_perr));
            fflush(stderr);
            GGML_ABORT("PXA_WAVE4_DIAG entry-dirty");
        }
    }
    // PXA_NODE_PROF per-graph wall summary
    int64_t _pxa_gt0 = 0;
    if (g_pxa_prof_on > 0) { cudaStreamSynchronize(cuda_ctx->stream()); _pxa_gt0 = ggml_time_us(); }
    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];

                if (ggml_is_noop(node)) continue;

                if (g_pxa_prof_on < 0) { g_pxa_prof_on = getenv("PXA_PROFILE") ? 1 : 0; if (getenv("PXA_PROFILE_EVERY")) g_pxa_prof_every = atol(getenv("PXA_PROFILE_EVERY")); if (g_pxa_prof_every < 1) g_pxa_prof_every = 20000; }
                int64_t _pxt0 = 0;
                if (g_pxa_prof_on > 0) { cudaStreamSynchronize(cuda_ctx->stream()); _pxt0 = ggml_time_us(); }
                pxa_dqc_walk(*cuda_ctx, cgraph, i, use_cuda_graph);
                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node, cgraph, i);
                if (!ok) {
                    GGML_CUDA_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
                if (g_pxa_prof_on > 0) { cudaStreamSynchronize(cuda_ctx->stream()); double _dus=(double)(ggml_time_us()-_pxt0); int _o=(int)node->op; if(_o>=0&&_o<128){ g_pxa_op_us[_o]+=_dus; g_pxa_op_cnt[_o]++; } { auto & pr = g_pxa_name_us[pxa_name_bucket(node->name)]; pr.first += _dus; pr.second++; } if ((++g_pxa_total_ops % g_pxa_prof_every)==0) pxa_prof_dump(); }
                // PXA_WAVE4_DIAG: sync+check after EVERY op -> pin the exact faulting node.
                if (g_pxa_op_check_on > 0) {
                    cudaError_t _perr = cudaStreamSynchronize(cuda_ctx->stream());
                    if (_perr == cudaSuccess) _perr = cudaGetLastError();
                    if (_perr != cudaSuccess) {
                        fprintf(stderr, "PXA_WAVE4_DIAG FAULT dev=%d node[%d/%d] name=%s op=%s err=%s\n",
                                cuda_ctx->device, i, cgraph->n_nodes, node->name, ggml_op_name(node->op), cudaGetErrorString(_perr));
                        ggml_tensor * _tt[1 + GGML_MAX_SRC]; _tt[0] = node;
                        for (int _s = 0; _s < GGML_MAX_SRC; ++_s) _tt[1+_s] = node->src[_s];
                        for (int _s = 0; _s < 1 + GGML_MAX_SRC; ++_s) {
                            ggml_tensor * _t = _tt[_s]; if (!_t) continue;
                            fprintf(stderr, "  %s name=%s type=%s ne=[%ld,%ld,%ld,%ld] nb=[%zu,%zu,%zu,%zu] data=%p bufbase=%p\n",
                                    _s == 0 ? "dst" : "src", _t->name, ggml_type_name(_t->type),
                                    (long)_t->ne[0], (long)_t->ne[1], (long)_t->ne[2], (long)_t->ne[3],
                                    _t->nb[0], _t->nb[1], _t->nb[2], _t->nb[3], _t->data,
                                    _t->buffer ? ggml_backend_buffer_get_base(_t->buffer) : (void *)0);
                        }
                        for (int _k = i >= 4 ? i - 4 : 0; _k < i; ++_k) {
                            fprintf(stderr, "  prior node[%d] name=%s op=%s\n", _k, cgraph->nodes[_k]->name, ggml_op_name(cgraph->nodes[_k]->op));
                        }
                        fflush(stderr);
                        GGML_ABORT("PXA_WAVE4_DIAG: pinned at node %s", node->name);
                    }
                }
            }
        }
#ifdef USE_CUDA_GRAPH
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }

            // PXA_CUDA_GRAPH_BATCH fail-soft (S2): a capture that hit an uncapturable operation
            // must not abort the process. Disable graphs for THIS key only and re-run the ggml graph
            // eagerly -- work recorded during capture was never executed, so the eager re-run is the
            // first (and only) execution. One-way: the key stays eager for the rest of the session.
            cudaError_t pxa_capture_err = cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph);
            {
                std::lock_guard<std::mutex> lock(ggml_cuda_lock);
                if (--ggml_cuda_lock_counter == 0) {
                    ggml_cuda_lock_cv.notify_all();
                }
            }
            if (pxa_capture_err != cudaSuccess || graph->graph == nullptr) {
                (void)cudaGetLastError(); // clear the sticky capture error
                GGML_CUDA_LOG_ERROR("%s: CUDA graph capture failed (%s) - falling back to eager for this graph\n",
                        __func__, cudaGetErrorString(pxa_capture_err));
                if (graph->graph != nullptr) {
                    (void)cudaGraphDestroy(graph->graph);
                    graph->graph = nullptr;
                }
                graph->disable_due_to_failed_graph_capture = true;
                use_cuda_graph = false;
                cuda_graph_update_required = false;
                pxa_cgraph_stats().disables_capture_fail++;
                if (pxa_cuda_graph_log_enabled()) {
                    fprintf(stderr, "PXA_CGRAPH DISABLE capture-fail dev=%d key=%p\n",
                            cuda_ctx->device, ggml_cuda_graph_get_key(cgraph));
                }
                continue; // re-run this ggml graph eagerly
            }
            graph->n_captures++;
            graph->last_n_nodes = cgraph->n_nodes;
            pxa_cgraph_stats().captures++;
            if (pxa_cuda_graph_log_enabled()) {
                int pxa_ny = -1; // Ny of the first MoE node in this (sub)graph, if any
                for (int j = 0; j < cgraph->n_nodes; ++j) {
                    if (cgraph->nodes[j]->op == GGML_OP_MOE_FUSED_UP_GATE) { pxa_ny = (int)cgraph->nodes[j]->src[2]->ne[2]; break; }
                }
                fprintf(stderr, "PXA_CGRAPH capture dev=%d nodes=%d moe_ny=%d key=%p captures=%ld replays=%ld consec=%d\n",
                        cuda_ctx->device, cgraph->n_nodes, pxa_ny, ggml_cuda_graph_get_key(cgraph),
                        graph->n_captures, graph->n_replays, graph->number_consecutive_updates);
                // LOG=2: full node list per capture -> diff two captures to NAME a topology delta.
                if (pxa_cuda_graph_log_level() >= 2) {
                    for (int j = 0; j < cgraph->n_nodes; ++j) {
                        const ggml_tensor * t = cgraph->nodes[j];
                        fprintf(stderr, "PXA_CGRAPH_NODE key=%p [%d] op=%s name=%s ne=%lld,%lld,%lld,%lld\n",
                                ggml_cuda_graph_get_key(cgraph), j, ggml_op_name(t->op), t->name,
                                (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
                    }
                }
            }
            graph_evaluated_or_captured = true; // CUDA graph has been captured
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }
    if (g_pxa_prof_on > 0 && _pxa_gt0 && cgraph->n_nodes >= 64) {
        cudaStreamSynchronize(cuda_ctx->stream());
        fprintf(stderr, "PXA_GRAPH dev=%d nodes=%d us=%lld\n", cuda_ctx->device, cgraph->n_nodes, (long long)(ggml_time_us() - _pxa_gt0));
    }

    if (use_cuda_graph) {
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        }
        if (cuda_graph_update_required) { // Update graph executable
            update_cuda_graph_executable(graph);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
        if (!cuda_graph_update_required) {
            // Honest replay accounting: a launch right after capture is NOT a replay.
            graph->n_replays++;
            long r = ++pxa_cgraph_stats().replays;
            if (pxa_cuda_graph_log_enabled()) {
                if (pxa_cuda_graph_log_level() >= 2) {
                    fprintf(stderr, "PXA_CGRAPH replay dev=%d nodes=%d key=%p key_replays=%ld total=%ld\n",
                            cuda_ctx->device, cgraph->n_nodes, ggml_cuda_graph_get_key(cgraph), graph->n_replays, r);
                } else if (r == 1 || r % 500 == 0) {
                    fprintf(stderr, "PXA_CGRAPH replays=%ld (captures=%ld)\n", r, pxa_cgraph_stats().captures.load());
                }
            }
        }
#else
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}


// ---------------------------------------------------------------------------------------
// PXA_SCHED_TIMELINE=<csv path>  (default OFF)
// Per-split device timeline: host wall time at enqueue plus GPU start/end taken with timing
// enabled cudaEvents on that device's own stream, all relative to a per-device base event.
// PXA_SCHED_TIMELINE_EPOCHS=N (default 6) dumps after N scheduler graphs and then stops.
// ---------------------------------------------------------------------------------------
struct pxa_tl_rec { int dev; int epoch; int split; int64_t host_us; cudaEvent_t ev0, ev1; };
static std::vector<pxa_tl_rec> pxa_tl_recs;
static cudaEvent_t pxa_tl_base[GGML_CUDA_MAX_DEVICES] = {};
static int         pxa_tl_epoch = -1;
static int         pxa_tl_split_ctr[GGML_CUDA_MAX_DEVICES] = {};
static int64_t     pxa_tl_host0 = 0;
static bool        pxa_tl_done = false;

static const char * pxa_tl_path(void) {
    static const char * p = getenv("PXA_SCHED_TIMELINE");
    return p;
}
static int pxa_tl_epochs(void) {
    static const int n = [] { const char * e = getenv("PXA_SCHED_TIMELINE_EPOCHS"); return e ? atoi(e) : 6; }();
    return n;
}

static void pxa_tl_flush(void) {
    if (pxa_tl_done || !pxa_tl_path()) return;
    pxa_tl_done = true;
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
        if (pxa_tl_base[d]) { ggml_cuda_set_device(d); cudaDeviceSynchronize(); }
    }
    FILE * f = fopen(pxa_tl_path(), "w");
    if (!f) return;
    fprintf(f, "dev,epoch,split,t_enqueue_host_us,t_start_gpu_us,t_end_gpu_us\n");
    for (auto & r : pxa_tl_recs) {
        float ms0 = 0.f, ms1 = 0.f;
        ggml_cuda_set_device(r.dev);
        cudaEventElapsedTime(&ms0, pxa_tl_base[r.dev], r.ev0);
        cudaEventElapsedTime(&ms1, pxa_tl_base[r.dev], r.ev1);
        fprintf(f, "%d,%d,%d,%lld,%.1f,%.1f\n", r.dev, r.epoch, r.split,
                (long long)(r.host_us - pxa_tl_host0), ms0 * 1000.0, ms1 * 1000.0);
    }
    fclose(f);
    fprintf(stderr, "PXA_SCHED_TIMELINE: wrote %zu splits to %s\n", pxa_tl_recs.size(), pxa_tl_path());
}

// ---------------------------------------------------------------------------------------
// PXA_SCHED_ASYNC_INPUTS: pinned staging ring for scheduler split inputs.
//
// The scheduler copies every host-resident split input with a BLOCKING cudaMemcpy on
// cudaStreamPerThread, and blocks the host on the destination device first so the source
// (llama's input buffers, rewritten by llama_set_inputs on the next ubatch) cannot be
// overwritten mid-copy. Both blocks are why only one card runs at a time.
//
// Staging removes the reason for both: the bytes are memcpy'd into a pinned buffer this
// module owns, and the device copy is issued on the destination device's COMPUTE stream,
// so it is ordered after that device's previous reads of the same slot with no host
// involvement at all. The pinned buffer is a ring of `gens` generations; a generation is
// reused only after the host has waited on the event recorded at the end of the graph that
// last wrote it, so the host may run up to `gens` graphs ahead of the GPUs.
// ---------------------------------------------------------------------------------------
struct pxa_stage_ent { const ggml_tensor * key; const void * src; size_t size; size_t off; };
struct pxa_stage_gen {
    char *      buf = nullptr;
    size_t      cap = 0;
    size_t      used = 0;
    std::vector<pxa_stage_ent> ents;
    cudaEvent_t      ev[GGML_CUDA_MAX_DEVICES]      = {};
    bool             ev_live[GGML_CUDA_MAX_DEVICES] = {};
    ggml_backend_t   be[GGML_CUDA_MAX_DEVICES]      = {};
};
static std::vector<pxa_stage_gen> pxa_stage_gens;
static int    pxa_stage_cur   = -1;
static size_t pxa_stage_req   = 0;   // bytes requested by the graph in flight
static size_t pxa_stage_hw    = 0;   // high-water over all graphs so far
static int64_t pxa_stage_hits = 0, pxa_stage_miss = 0, pxa_stage_alias = 0;

// Must stay byte-for-byte the same rule as pxa_sched_async_inputs() in ggml-backend.cpp:
// if the scheduler thinks inputs are staged and this module does not (or the reverse), the
// scheduler skips a guard for a copy that never got staged.
static bool pxa_stage_on(void) {   // default OFF; PXA_SCHED_ASYNC_INPUTS=1, or PXA_SCHED_PIPELINE=1, arms it
    static const bool v = [] {
        const char * e = getenv("PXA_SCHED_ASYNC_INPUTS");
        if (e != nullptr) {
            return atoi(e) != 0;
        }
        const char * p = getenv("PXA_SCHED_PIPELINE");
        return p != nullptr && atoi(p) != 0;
    }();
    return v;
}
static int pxa_stage_ngens(void) {
    static const int n = [] { const char * e = getenv("PXA_SCHED_STAGE_GENS"); int v = e ? atoi(e) : 3; return v < 1 ? 1 : (v > 8 ? 8 : v); }();
    return n;
}
static size_t pxa_stage_cap_max(void) {
    static const size_t n = [] { const char * e = getenv("PXA_SCHED_STAGE_MAX_MB"); return (size_t)(e ? atoi(e) : 640) << 20; }();
    return n;
}

extern "C" void ggml_cuda_stage_graph_begin(void) {
    if (!pxa_stage_on()) return;
    if (pxa_stage_gens.empty()) pxa_stage_gens.resize(pxa_stage_ngens());
    pxa_stage_cur = (pxa_stage_cur + 1) % (int) pxa_stage_gens.size();
    pxa_stage_gen & g = pxa_stage_gens[pxa_stage_cur];
    // the graph that last wrote this generation must be done reading it out of host memory
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
        if (g.ev_live[d]) {
            ggml_cuda_set_device(d);
            CUDA_CHECK(cudaEventSynchronize(g.ev[d]));
            g.ev_live[d] = false;
        }
    }
    // grow to the high water, capped: a graph that wants more than the cap still stages the
    // first cap_max bytes and falls back for the rest, instead of staging nothing at all
    const size_t want_cap = pxa_stage_hw < pxa_stage_cap_max() ? pxa_stage_hw : pxa_stage_cap_max();
    if (g.cap < want_cap) {
        if (g.buf) CUDA_CHECK(cudaFreeHost(g.buf));
        g.buf = nullptr; g.cap = 0;
        void * p = nullptr;
        if (cudaHostAlloc(&p, want_cap, cudaHostAllocPortable) == cudaSuccess) {
            g.buf = (char *) p;
            g.cap = want_cap;
        } else {
            cudaGetLastError();
        }
    }
    g.used = 0;
    g.ents.clear();
    pxa_stage_req = 0;
    // firing proof, and the running staged/fallback tally so an A/B can never be run on a
    // binary where the lever silently did nothing
    static int64_t n_graphs = 0;
    if (++n_graphs == 1 || (n_graphs & 63) == 0) {
        fprintf(stderr, "PXA_SCHED_ASYNC_INPUTS: graph %lld gens=%d cap=%zu MiB staged=%lld fallback=%lld alias=%lld\n",
                (long long) n_graphs, (int) pxa_stage_gens.size(), g.cap >> 20,
                (long long) pxa_stage_hits, (long long) pxa_stage_miss, (long long) pxa_stage_alias);
    }
}

extern "C" void ggml_cuda_stage_graph_end(void) {
    if (!pxa_stage_on() || pxa_stage_cur < 0) return;
    pxa_stage_gen & g = pxa_stage_gens[pxa_stage_cur];
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
        if (g.be[d] == nullptr) continue;
        ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) g.be[d]->context;
        ggml_cuda_set_device(d);
        if (g.ev[d] == nullptr) CUDA_CHECK(cudaEventCreateWithFlags(&g.ev[d], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(g.ev[d], ctx->stream()));
        g.ev_live[d] = true;
        g.be[d] = nullptr;
    }
    if (pxa_stage_req > pxa_stage_hw) pxa_stage_hw = pxa_stage_req;
}

// Stage `size` bytes of host memory at `src` and issue the host->device copy into
// `dst` + `offset` on the destination backend's compute stream. Returns false when the
// bytes do not fit this generation; the caller then falls back to the blocking path.
extern "C" bool ggml_cuda_stage_h2d(ggml_backend_t backend, ggml_tensor * dst, const ggml_tensor * key, const void * src, size_t offset, size_t size) {
    if (!pxa_stage_on() || pxa_stage_cur < 0 || size == 0) return false;
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    const int dev = ctx->device;
    if (dev < 0 || dev >= GGML_CUDA_MAX_DEVICES) return false;
    pxa_stage_gen & g = pxa_stage_gens[pxa_stage_cur];

    // The cache exists so that ONE source tensor fanning out to both devices is memcpy'd
    // once. It must therefore key on the tensor, not on its host address: ggml-alloc reuses
    // a freed range inside the same graph, so two different split inputs can share an
    // address, and an address-keyed hit hands the second one the first one's bytes. That is
    // a silent content swap whose reachability depends on the allocation layout - i.e. on
    // exactly the things (PXA_DN_SHARE_INPUTS, ubatch size) the corruption tracked.
    size_t off = SIZE_MAX;
    bool   known = false;
    for (const auto & e : g.ents) {
        if (e.key == key && e.src == src && e.size == size) { off = e.off; known = true; break; }
        if (e.src == src && e.size == size) {
            // same host range, different tensor: the address-keyed cache would have hit here
            pxa_stage_alias++;
            if (pxa_stage_alias == 1) {
                fprintf(stderr, "PXA_SCHED_ASYNC_INPUTS: host range %p (%zu bytes) is shared by two "
                        "distinct split inputs in one graph (%s / %s) - staging them separately\n",
                        src, size, e.key && e.key->name[0] ? e.key->name : "?", key && key->name[0] ? key->name : "?");
            }
        }
    }
    if (!known) {
        pxa_stage_req += (size + 255) & ~(size_t)255;
        const size_t base = (g.used + 255) & ~(size_t)255;
        if (g.buf == nullptr || base + size > g.cap) {
            g.ents.push_back({key, src, size, SIZE_MAX});   // remember, so the request is counted once
        } else {
            memcpy(g.buf + base, src, size);
            off    = base;
            g.used = base + size;
            g.ents.push_back({key, src, size, off});
        }
    }
    if (off == SIZE_MAX) { pxa_stage_miss++; return false; }

    ggml_cuda_set_device(dev);
    CUDA_CHECK(cudaMemcpyAsync((char *) dst->data + offset, g.buf + off, size, cudaMemcpyHostToDevice, ctx->stream()));
    g.be[dev] = backend;
    pxa_stage_hits++;
    return true;
}

extern "C" void ggml_cuda_stage_report(void) {
    if (!pxa_stage_on()) return;
    fprintf(stderr, "PXA_SCHED_ASYNC_INPUTS: staged=%lld fallback=%lld alias=%lld gen_cap=%zu MiB x%d\n",
            (long long) pxa_stage_hits, (long long) pxa_stage_miss, (long long) pxa_stage_alias,
            pxa_stage_gens.empty() ? 0 : pxa_stage_gens[0].cap >> 20, (int) pxa_stage_gens.size());
}

extern "C" void ggml_cuda_timeline_epoch(void) {
    if (!pxa_tl_path() || pxa_tl_done) return;
    ++pxa_tl_epoch;
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) pxa_tl_split_ctr[d] = 0;
    if (pxa_tl_epoch >= pxa_tl_epochs()) pxa_tl_flush();
}

GGML_CALL static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    // PXA_SCHED_TIMELINE: bracket this split's enqueue on the split device's own stream
    ggml_backend_cuda_context * pxa_tl_ctx = (ggml_backend_cuda_context *)backend->context;
    const bool pxa_tl_rec_on = pxa_tl_path() && !pxa_tl_done && pxa_tl_epoch >= 0 && pxa_tl_recs.size() < 4096;
    cudaEvent_t pxa_tl_e0 = nullptr, pxa_tl_e1 = nullptr;
    int64_t pxa_tl_t0 = 0;
    int pxa_tl_dev = pxa_tl_ctx->device;
    if (pxa_tl_rec_on) {
        ggml_cuda_set_device(pxa_tl_dev);
        if (!pxa_tl_base[pxa_tl_dev]) {
            cudaEventCreate(&pxa_tl_base[pxa_tl_dev]);
            cudaEventRecord(pxa_tl_base[pxa_tl_dev], pxa_tl_ctx->stream());
        }
        if (pxa_tl_host0 == 0) pxa_tl_host0 = ggml_time_us();
        cudaEventCreate(&pxa_tl_e0);
        cudaEventCreate(&pxa_tl_e1);
        pxa_tl_t0 = ggml_time_us();
        cudaEventRecord(pxa_tl_e0, pxa_tl_ctx->stream());
    }

    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    // PXA_MTP_REDUCE_CTX_FIX: ggml_cuda_op_reduce synchronizes its cross-device peer copies using
    // the GLOBAL ggml_cuda_info().all_ctx[i] streams/events. all_ctx[i] is set ONLY by the FIRST
    // constructed context for device i (the ctor refuses to overwrite and logs a-context-already
    // -exists), so once a SECOND llama_context exists (the MTP speculative/draft context), its
    // reduces gather the other-device partials on the WRONG first/main context streams. The
    // producer kernel ran on THIS context device-i stream while the gather waits on the other
    // context stream -> unordered -> intermittent cross-device illegal memory access in the MTP
    // head decode under -sm attn (crash device varies). Re-register THIS context for its device on
    // every graph compute: the scheduler runs each device splits host-sequentially before any later
    // cross-device reduce, so by the reduce all participating all_ctx[] point to the CURRENT context.
    // Single-context (no-MTP) decode is unaffected (idempotent no-op).
    {
        auto _pxa_info = const_cast<ggml_cuda_device_info*>(&ggml_cuda_info());
        if (cuda_ctx->device >= 0 && cuda_ctx->device < GGML_CUDA_MAX_DEVICES) {
            _pxa_info->all_ctx[cuda_ctx->device] = cuda_ctx;
        }
    }

#ifdef USE_CUDA_GRAPH
    cuda_ctx->cur_graph = nullptr;

    static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);

    // Disable CUDA graphs in presence of env var, old GPU, use-case which is changing too rapidly,
    // or previous graph capture failure.
    // Also disable for multi-gpu for now. TO DO investigate
    bool use_cuda_graph = !disable_cuda_graphs_due_to_env && cuda_ctx->use_cuda_graph;

    ggml_cuda_graph * graph = nullptr;
    if (use_cuda_graph) {
        auto graph_key = ggml_cuda_graph_get_key(cgraph);
        graph = ggml_cuda_get_graph(*cuda_ctx, graph_key);
    }
    cuda_ctx->cur_graph = graph;

    bool cuda_graph_update_required = false;

    const bool pxa_v2 = pxa_cuda_graph_v2_enabled();
    const bool pxa_prefill = pxa_cuda_graph_mainline_policy();
    const bool pxa_glog = pxa_cuda_graph_log_enabled();
    if (pxa_glog) pxa_cgraph_stats_arm_atexit();
    const void * pxa_gkey = (pxa_glog && graph) ? ggml_cuda_graph_get_key(cgraph) : nullptr;

    if (use_cuda_graph && graph->graph == nullptr) {
        // PXA_CUDA_GRAPH_PREFILL: mainline's gate is cc < CC_VOLTA; ours was cc < CC_AMPERE. With the
        // flag on we lift it to cc < CC_PASCAL (i.e. allow both P100 and V100) and let the measurement
        // decide, since stream capture itself is supported from sm_60 on.
        // PXA_CUDA_GRAPH_PREFILL lifts the gate to sm_60 (it is a P100+V100 experiment);
        // PXA_CUDA_GRAPH_DECODE alone only needs sm_70. Neither value is a hardware limit --
        // stream capture works from sm_60 -- both are policy set by measurement.
        const int pxa_graph_min_cc = pxa_cuda_graph_prefill_enabled() ? CC_PASCAL
                                   : (pxa_cuda_graph_decode_enabled() ? CC_VOLTA : CC_AMPERE);
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < pxa_graph_min_cc && !getenv("PXA_CUDA_GRAPHS_PASCAL")) {
            if (!graph->disable_due_to_gpu_arch) {
                pxa_cgraph_stats().disables_arch++;
                if (pxa_glog) fprintf(stderr, "PXA_CGRAPH DISABLE gpu-arch dev=%d key=%p\n", cuda_ctx->device, pxa_gkey);
            }
            graph->disable_due_to_gpu_arch = true;
#ifndef NDEBUG
            GGML_CUDA_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
#endif
        }
    }

    if (use_cuda_graph && (
        graph->disable_due_to_gpu_arch ||
        graph->disable_due_to_too_many_updates ||
        graph->disable_due_to_failed_graph_capture)) {
        use_cuda_graph = false;
        graph->n_eager++;
        pxa_cgraph_stats().eager_disabled++;
    }

    // PXA_CUDA_GRAPH_V2 P1.2: cooldown in place of the permanent too-many-updates disable.
    if (use_cuda_graph && pxa_v2 && graph->cooldown_remaining > 0) {
        graph->cooldown_remaining--;
        if (graph->cooldown_remaining == 0) {
            graph->number_consecutive_updates = 0; // re-arm
            if (pxa_glog) fprintf(stderr, "PXA_CGRAPH cooldown-rearm dev=%d key=%p\n", cuda_ctx->device, pxa_gkey);
        }
        use_cuda_graph = false;
        graph->n_eager++;
        pxa_cgraph_stats().eager_cooldown++;
    }

    if (use_cuda_graph && (pxa_v2 || pxa_prefill)) {
        // PXA_CUDA_GRAPH_V2 P1.3: capturability check BEFORE the property store, so an uncapturable
        // graph (prefill MoE chunk, unsupported CPY) can no longer clobber the stored properties of
        // a capturable one at the same key and force a spurious recapture of the next decode token.
        use_cuda_graph = check_node_graph_compatibility_and_refresh_copy_ops(cuda_ctx, graph, cgraph, use_cuda_graph, cuda_ctx->stream());
        if (!use_cuda_graph) {
            graph->n_eager++;
            pxa_cgraph_stats().eager_uncapturable++;
            if (pxa_cuda_graph_log_level() >= 2) {
                fprintf(stderr, "PXA_CGRAPH eager-uncapturable dev=%d nodes=%d key=%p\n", cuda_ctx->device, cgraph->n_nodes, pxa_gkey);
            }
        } else {
            bool pxa_props_changed = false;
            cuda_graph_update_required = is_cuda_graph_update_required(graph, cgraph, pxa_gkey,
                                                                       pxa_prefill ? &pxa_props_changed : nullptr);

            // P3 pool-generation guard: if pool memory was cudaFree'd since the last capture, the
            // captured kernels may hold dangling scratch pointers -> force recapture, never replay.
            const uint64_t pxa_pool_gen = cuda_ctx->pool().generation();
            if (!cuda_graph_update_required && pxa_pool_gen != graph->pool_generation) {
                cuda_graph_update_required = true;
                if (pxa_glog) fprintf(stderr, "PXA_CGRAPH update-required key=%p reason=pool-generation %llu->%llu\n",
                        pxa_gkey, (unsigned long long)graph->pool_generation, (unsigned long long)pxa_pool_gen);
            }
            graph->pool_generation = pxa_pool_gen;

            if (cuda_graph_update_required) {
                graph->number_consecutive_updates++;
            } else {
                graph->number_consecutive_updates = 0;
            }

            if (pxa_prefill) {
                // PXA_CUDA_GRAPH_PREFILL: mainline 9400c8946e admission rule, in place of the
                // consecutive-update disable. Capture only once the node properties have been
                // IDENTICAL on two consecutive evals; after that, any property change reverts to
                // eager (re-arming the warmup) instead of paying a re-capture we would throw away on
                // the next eval. A prefill that walks n_kv up 256 cells per ubatch therefore never
                // captures and costs only the property compare; one that settles on a shape captures
                // once and replays for every later ubatch of that shape.
                if (!graph->warmup_complete) {
                    if (!pxa_props_changed) {
                        graph->warmup_complete = true;
                        // cuda_graph_update_required stays as computed: true iff there is no
                        // executable yet (first capture) or the pool generation moved.
                        if (pxa_glog) fprintf(stderr, "PXA_CGRAPH warmup-complete dev=%d key=%p nodes=%d\n",
                                cuda_ctx->device, pxa_gkey, cgraph->n_nodes);
                    } else {
                        use_cuda_graph = false; // still churning: run eager, capture nothing
                        cuda_graph_update_required = false;
                        cuda_ctx->cur_graph = nullptr;
                        graph->n_eager++;
                        pxa_cgraph_stats().eager_cooldown++;
                    }
                } else if (pxa_props_changed) {
                    // Post-warmup property change: back to eager and re-arm, never a blind recapture.
                    graph->warmup_complete = false;
                    use_cuda_graph = false;
                    cuda_graph_update_required = false;
                    cuda_ctx->cur_graph = nullptr;
                    graph->n_eager++;
                    pxa_cgraph_stats().eager_cooldown++;
                    if (pxa_glog) fprintf(stderr, "PXA_CGRAPH warmup-reset dev=%d key=%p\n", cuda_ctx->device, pxa_gkey);
                }
            } else if (graph->number_consecutive_updates >= 4) {
                const int rearm = pxa_cuda_graph_rearm_evals();
                if (rearm > 0) {
                    graph->cooldown_remaining = rearm;
                    pxa_cgraph_stats().cooldowns_armed++;
                    if (pxa_glog) fprintf(stderr, "PXA_CGRAPH cooldown-armed dev=%d key=%p evals=%d\n", cuda_ctx->device, pxa_gkey, rearm);
                } else {
                    graph->disable_due_to_too_many_updates = true; // PXA_CUDA_GRAPH_REARM=0 keeps the permanent disable
                    pxa_cgraph_stats().disables_too_many++;
                    if (pxa_glog) fprintf(stderr, "PXA_CGRAPH DISABLE too-many-updates dev=%d key=%p\n", cuda_ctx->device, pxa_gkey);
                }
                use_cuda_graph = false;
                cuda_ctx->cur_graph = nullptr;
                graph->n_eager++;
            }
        }
    } else if (use_cuda_graph) {
        cuda_graph_update_required = is_cuda_graph_update_required(graph, cgraph, pxa_gkey);

        use_cuda_graph = check_node_graph_compatibility_and_refresh_copy_ops(cuda_ctx, graph, cgraph, use_cuda_graph, cuda_ctx->stream());
        if (!use_cuda_graph) {
            graph->n_eager++;
            pxa_cgraph_stats().eager_uncapturable++;
        }

        // Disable CUDA graphs (from the next token) if the use-case is demanding too many consecutive graph updates.
        if (use_cuda_graph) {
            if (cuda_graph_update_required) {
                graph->number_consecutive_updates++;
            } else {
                graph->number_consecutive_updates = 0;
            }
        }

        if (graph->number_consecutive_updates >= 4) {
            if (!graph->disable_due_to_too_many_updates) {
                pxa_cgraph_stats().disables_too_many++;
                if (pxa_glog) fprintf(stderr, "PXA_CGRAPH DISABLE too-many-updates dev=%d key=%p\n", cuda_ctx->device, pxa_gkey);
            }
            graph->disable_due_to_too_many_updates = true;
            use_cuda_graph = false;
            cuda_ctx->cur_graph = nullptr;
#ifndef NDEBUG
            GGML_CUDA_LOG_DEBUG("%s: disabling CUDA graphs due to too many consecutive updates\n", __func__);
#endif
        }
    }

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        // Why are we protecting an atomic_int with a mutex?
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ++ggml_cuda_lock_counter;
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    if (graph && !use_cuda_graph) {
        graph->use_cpy_indirection = false;
    }

#else
    bool use_cuda_graph = false;
    bool cuda_graph_update_required = false;
#endif // USE_CUDA_GRAPH

    bool graph_evaluated_or_captured = false;

    evaluate_and_capture_cuda_graph(cuda_ctx, cgraph, graph_evaluated_or_captured, use_cuda_graph, cuda_graph_update_required);

    if (pxa_tl_rec_on) {
        ggml_cuda_set_device(pxa_tl_dev);
        cudaEventRecord(pxa_tl_e1, pxa_tl_ctx->stream());
        pxa_tl_recs.push_back({pxa_tl_dev, pxa_tl_epoch, pxa_tl_split_ctr[pxa_tl_dev]++, pxa_tl_t0, pxa_tl_e0, pxa_tl_e1});
    }
    return GGML_STATUS_SUCCESS;
}

GGML_CALL static bool ggml_backend_cuda_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    // Non-mul_mat ops can't read a split-buffer parent (no data ptr); let the scheduler fall back to CPU.
    if (op->op != GGML_OP_MUL_MAT && op->op != GGML_OP_MUL_MAT_ID) {
        // PXA-SHARD: the fused MoE up+gate op HAS a CUDA handler that reads the
        // expert-shard per-device slices (ggml_cuda_moe_up_gate_unary's shard branch
        // resolves root->name->registry), so it must STAY on CUDA — do NOT fall it
        // back to CPU for an expert-shard src. (Split buffers still always fall back.)
        const bool pxa_shard_fused_ok = (op->op == GGML_OP_MOE_FUSED_UP_GATE || op->op == GGML_OP_FUSED_UP_GATE);
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            // no single data ptr for split/shard bufts, so non-mul_mat ops fall back
            // to CPU. Both terms are false when PXA_EXPERT_SHARD is off.
            if (op->src[i] && op->src[i]->buffer &&
                ggml_backend_buft_is_cuda_split(op->src[i]->buffer->buft)) {
                return false;
            }
            if (op->src[i] && op->src[i]->buffer &&
                pxa_buft_is_expert_shard(op->src[i]->buffer->buft) && !pxa_shard_fused_ok) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_SWIGLU:
                case GGML_UNARY_OP_SWIGLU_OAI:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_FUSED_MUL_UNARY: return ggml_is_contiguous(op->src[0]);
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_MOE_FUSED_UP_GATE:
        case GGML_OP_FUSED_UP_GATE:
            {
                bool is_fused_up_gate = op->op == GGML_OP_MOE_FUSED_UP_GATE || op->op == GGML_OP_FUSED_UP_GATE;
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = is_fused_up_gate ? op->src[2] : op->src[1];
                if (is_fused_up_gate && op->src[1] && !ggml_moe_up_gate_can_fuse(a->type, op->src[1]->type)) {
                    fprintf(stderr, "%s: returning false for GGML_OP_MOE_FUSED_UP_GATE because src0->type != src1->type\n", __func__);
                    return false;
                }
                //==================================================================
                //if (ggml_is_quantized(a->type) && ggml_is_quantized(b->type)) {
                //    return false;
                //}
                //==================================================================
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16 && !ggml_is_quantized(a->type)) {
                    printf("%s: returning false for op %d because (case 1)\n", __func__, (int)op->op);
                    return false;
                }
                if (op->op == GGML_OP_MUL_MAT && a->ne[3] != b->ne[3]) {
                    return false;
                }
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q6_0:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_PXQ4:
                    case GGML_TYPE_PXQ4HQ:
                    case GGML_TYPE_PXQ2:      // ADD
                    case GGML_TYPE_PXQ3:      // ADD
                    case GGML_TYPE_PXQ1:
                    case GGML_TYPE_PXQ6:     // ADD
                    case GGML_TYPE_IQ4_XS:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_SET_ROWS:
            {
                return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                       op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                       op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                       op->src[0]->type == GGML_TYPE_F32 &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_BF16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q6_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    // Same-type integer copy -- a strided byte copy, nothing to convert.
                    // The implementation in cpy.cu must agree with this gate: DeepSeek-V4's
                    // hash-routed blocks produce an I32 expert selection whose get_rows has no
                    // CUDA path, the scheduler splits there, and the resulting I32 move aborted
                    // the backend at ~3.8k prompt tokens.
                    return true;
                }
                if (ggml_is_quantized(src0_type) && (src1_type == GGML_TYPE_F16 || src1_type == GGML_TYPE_F32)) {
                    return true;
                }
                if (ggml_is_contiguous(op->src[0]) && ggml_are_same_shape(op->src[0], op->src[1])) {
                    if (src1_type == GGML_TYPE_F16 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F32) {
                        return true;
                    }
                }
                if (ggml_are_same_shape(op->src[0], op->src[1]) && op->src[0]->type == GGML_TYPE_Q8_0 && op->src[1]->type == GGML_TYPE_Q8_0) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_REDUCE:
        case GGML_OP_FAKE_CPY:
        case GGML_OP_ARGMAX:
            return true;
        case GGML_OP_HADAMARD: {
            if (!(op->op_params[0] == 64 || op->op_params[0] == 128 || op->op_params[0] == 256 || op->op_params[0] == 512)) return false;
            if (op->ne[0] % op->op_params[0] != 0) return false;
            if (op->type != GGML_TYPE_F32) return false;
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q6_0:
                case GGML_TYPE_IQ4_NL:
                    return true;
                default:
                    return false;
            }
        }
        case GGML_OP_DUP:
        case GGML_OP_REPEAT:
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            return true;
        case GGML_OP_L2_NORM:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]) && op->ne[0] % WARP_SIZE == 0;
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_MULTI_ADD:
        case GGML_OP_MUL_MULTI_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SUB:
        case GGML_OP_FUSED_RMS_NORM:
        case GGML_OP_FUSED_RMS_RMS_ADD:
        case GGML_OP_SCALE:
        case GGML_OP_SOFTCAP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_CLAMP:
        case GGML_OP_CONT:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SOFT_CAP_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ROPE_FAST:
        case GGML_OP_ROPE_CACHE:
            return true;
        case GGML_OP_FUSED_NORM:
            return ggml_is_contiguous(op->src[0]);
        //case GGML_OP_ROPE:
        //    return ggml_is_contiguous(op->src[0]);
        case GGML_OP_IM2COL:
        case GGML_OP_POOL_2D:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ARGSORT_THRESH:
        case GGML_OP_GROUPED_TOPK:
        case GGML_OP_ACC:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
            return true;
        case GGML_OP_CUMSUM:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_TRI:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[0]->type == op->type;
        case GGML_OP_FILL:
            return ggml_is_contiguous(op) && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        case GGML_OP_SOLVE_TRI:
            return ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   ggml_is_contiguous(op) &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   op->src[0]->ne[0] == op->src[0]->ne[1] &&
                   op->src[0]->ne[1] == op->src[1]->ne[1] &&
                   op->src[0]->ne[2] == op->src[1]->ne[2] &&
                   op->src[0]->ne[3] == op->src[1]->ne[3];
        case GGML_OP_SSM_CONV:
            return op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->src[2]->type == GGML_TYPE_F32 &&
                   op->src[3]->type == GGML_TYPE_I32 &&
                   op->type == GGML_TYPE_F32 &&
                   op->src[0]->nb[0] == sizeof(float) &&
                   op->src[1]->nb[0] == sizeof(float) &&
                   op->src[2]->nb[0] == sizeof(float) &&
                   op->src[3]->nb[0] == sizeof(int32_t) &&
                   op->src[2]->ne[0] == op->src[0]->ne[0] + 1 &&
                   op->src[2]->ne[1] == op->src[0]->ne[1] &&
                   op->src[1]->ne[0] == op->src[0]->ne[1] &&
                   op->src[3]->ne[0] == op->src[0]->ne[2];
        case GGML_OP_DELTA_NET:
            return true;
        case GGML_OP_DSV4_HC_SPLIT_SINKHORN:
        case GGML_OP_DSV4_HC_WEIGHTED_SUM:
        case GGML_OP_DSV4_HC_EXPAND:
            // the gate and the implementation must agree (the I32->I32 cpy lesson):
            // the kernels are all-F32
            return op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_MASK_TO_IDX:
            // the gate and the implementation must agree (the I32->I32 cpy lesson):
            // k_mask_to_idx handles F16/F32 masks only, writes I32, and indexes both
            // tensors' dim 0 as a dense array.
            return op->type == GGML_TYPE_I32 &&
                   (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->nb[0] == sizeof(int32_t) &&
                   op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) &&
                   op->src[0]->ne[0] >= op->ne[0] &&
                   op->src[0]->ne[1] == op->ne[1] &&
                   op->src[0]->ne[2] == op->ne[2] &&
                   op->src[0]->ne[3] == op->ne[3];
        case GGML_OP_FLASH_ATTN_EXT:
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
            return (op->src[0]->ne[0] == 64 && op->src[1]->type == GGML_TYPE_F16) || op->src[0]->ne[0] == 128;
#else
            return ggml_cuda_fattn_is_supported(*cuda_ctx, op);
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
        default:
            return false;
    }

    GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_cuda_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    //printf("%s(%s, %s): %d, %d\n", __func__, ggml_backend_name(backend), ggml_backend_buft_name(buft), ggml_backend_buft_is_cuda_split(buft), ggml_backend_buft_is_cuda(buft));
    if (ggml_backend_buft_is_cuda_split(buft)) {
        return true;
    }

    // PXA-SHARD: keep expert-sharded weights on the CUDA backend (same as split).
    // Always false when PXA_EXPERT_SHARD is off, so this is a no-op flag-off.
    if (pxa_buft_is_expert_shard(buft)) {
        return true;
    }

    if (ggml_backend_buft_is_cuda(buft)) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
        ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;
        return buft_ctx->device == cuda_ctx->device;
    }

    return false;
}

GGML_CALL static bool ggml_backend_cuda_offload_op(ggml_backend_t backend, const ggml_tensor * op) {
    auto ctx = (ggml_backend_cuda_context *)backend->context;
    int min_batch_size = ctx->offload_batch_size; // originally: GGML_CUDA_MIN_BATCH_OFFLOAD;

    // Why do we want to do this? The heuristics that the batch must have more than min_batch_size tokens to be worth it
    // offloading the required model weights comes from dense models. For MoE models, the average number of tokens
    // each expert deals with in a batch is (active_experts / total_experts) * batch_size. Hence, according to the
    // learned heuristics, we need (active_experts / total_experts) * batch_size >= min_batch_size.
    // Rearranging we get
    //
    //           batch_size * active_experts >= min_batch_size * total_experts
    //
    // as the condition for offloading model weights resinding in RAM to the GPU.
    // In this case, the number of tokens is not as usual in op->ne[1] but rather in op->ne[2].
    if (op->op == GGML_OP_MUL_MAT_ID || op->op == GGML_OP_MOE_FUSED_UP_GATE) {
        if (ctx->offload_batch_size_per_byte >= 0) {
            auto src0 = op->src[0];
            auto row_size = ggml_row_size(src0->type, src0->ne[0]);
            min_batch_size = int(1.*ctx->offload_batch_size_per_byte*row_size/src0->ne[0]);
        }
        auto ids = op->op == GGML_OP_MUL_MAT_ID ? op->src[2] : op->src[3];
        int64_t batch_size = op->ne[2];
        if (batch_size < min_batch_size) return false;
        int64_t n_experts_tot    = op->src[0]->ne[2];
        int64_t n_experts_active = ids->ne[0];
        bool should_offload = batch_size*n_experts_active >= min_batch_size*n_experts_tot;
        return should_offload;
    }

    return op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS;

    // Original:
    //return (op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS) ||
    //       (op->ne[2] >= min_batch_size && (op->op == GGML_OP_MUL_MAT_ID || op->op == GGML_OP_MOE_FUSED_UP_GATE));

    GGML_UNUSED(backend);
}

static ggml_backend_event_t ggml_backend_cuda_event_new(ggml_backend_t backend) {
#ifdef GGML_CUDA_NO_PEER_COPY
    return nullptr;
#else
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .backend = */ backend,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_event_free(ggml_backend_event_t event) {
    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));

    delete event;
}

static void ggml_backend_cuda_event_record(ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)event->backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(event->backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_event_synchronize(ggml_backend_event_t event) {
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .get_default_buffer_type = */ ggml_backend_cuda_get_default_buffer_type,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .supports_op             = */ ggml_backend_cuda_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_offload_op,
    /* .event_new               = */ ggml_backend_cuda_event_new,
    /* .event_free              = */ ggml_backend_cuda_event_free,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .event_synchronize       = */ ggml_backend_cuda_event_synchronize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

struct cuda_params {
    int  fusion = GGML_CUDA_FUSION;
    int  offload_batch_size = GGML_CUDA_MIN_BATCH_OFFLOAD;
    int  offload_batch_size_per_byte = -1;
    int  mmq_id_thresh = 32;
    float fa_offset = 0.6931f;
#ifdef USE_CUDA_GRAPH
    bool use_cuda_graph = true;
#else
    bool use_cuda_graph = false;
#endif
    bool enable_p2p = true;
};

static std::vector<std::string> string_split(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

template <typename T> bool read_value(const std::string& val, T& result) {
    std::istringstream str(val);
    T tmp; str >> tmp;
    if (!str.fail()) {
        result = tmp;
        return true;
    }
    return false;
}

static cuda_params ggml_cuda_parse_params(const char * params_string) {
    cuda_params params{};
    if (!params_string) return params;
    auto values = string_split(std::string{params_string}, ",");
    if (values.empty()) return params;
    for (auto& value : values) {
        auto parsed = string_split(value, "=");
        bool is_good = false;
        if (parsed.size() == 2) {
            if (parsed[0] == "fusion") {
                is_good = read_value(parsed[1], params.fusion);
            }
            else if (parsed[0] == "offload-batch-size") {
                int tmp = 0;
                is_good = read_value(parsed[1], tmp);
                if (is_good) {
                    params.offload_batch_size = tmp;
                }
            }
            else if (parsed[0] == "offload-batch-size-per-byte") {
                is_good = read_value(parsed[1], params.offload_batch_size_per_byte);
            }
            else if (parsed[0] == "mmq-id-size") {
                is_good = read_value(parsed[1], params.mmq_id_thresh);
            }
            else if (parsed[0] == "enable-p2p") {
                is_good = read_value(parsed[1], params.enable_p2p);
            }
            else if (parsed[0] == "fa-offset") {
                float tmp;
                is_good = read_value(parsed[1], tmp);
                if (is_good) {
                    if (tmp < 0.0f || tmp > 3.0f) {
                        GGML_CUDA_LOG_WARN("%s: bad value for %s. It is %g, but must be in [0...3]\n", __func__, parsed[0].c_str(), tmp);
                    } else {
                        params.fa_offset = tmp;
                    }
                }
            }
#ifdef USE_CUDA_GRAPH
            else if (parsed[0] == "graphs") {
                is_good = read_value(parsed[1], params.use_cuda_graph);
            }
#endif
        }
        if (!is_good) {
            GGML_CUDA_LOG_WARN("%s: invalid parameter %s (%d) -> ignored\n", __func__, value.c_str(), (int)parsed.size());
        }
    }
    return params;
}

GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device, [[maybe_unused]] const void * param_string) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_CUDA_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_CUDA_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_cuda_guid(),
        /* .interface = */ ggml_backend_cuda_interface,
        /* .context   = */ ctx
    };

    bool enable_p2p = true;
    if (param_string) {
        [[maybe_unused]] auto params = ggml_cuda_parse_params((const char *)param_string);
        if (params.fusion != ctx->fusion) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting fusion to %d\n", __func__, params.fusion);
            ctx->fusion             = params.fusion;
        }
        if (params.offload_batch_size != ctx->offload_batch_size) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting offload_batch_size to %d\n", __func__, params.offload_batch_size);
            ctx->offload_batch_size = params.offload_batch_size;
        }
        if (params.offload_batch_size_per_byte != ctx->offload_batch_size_per_byte) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting offload_batch_size_per_byte to %d\n", __func__, params.offload_batch_size_per_byte);
            ctx->offload_batch_size_per_byte = params.offload_batch_size_per_byte;
        }
        if (params.mmq_id_thresh != ctx->mmq_id_thresh) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting mmq_id_thresh to %d\n", __func__, params.mmq_id_thresh);
            ctx->mmq_id_thresh      = params.mmq_id_thresh;
        }
        if (params.fa_offset != ctx->fa_offset) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting fa_offset to %g\n", __func__, params.fa_offset);
            ctx->fa_offset = params.fa_offset;
        }
        enable_p2p = params.enable_p2p;
#ifdef USE_CUDA_GRAPH
        if (params.use_cuda_graph != ctx->use_cuda_graph) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting use_cuda_graph to %d\n", __func__, params.use_cuda_graph);
            ctx->use_cuda_graph = params.use_cuda_graph;
        }
#endif
    }

#ifdef GGML_USE_NCCL
    if (!enable_p2p) {
        printf("================== P2P disabled, but needed for NCCL\n");
        enable_p2p = true;
    }
#endif

#if !defined(GGML_CUDA_NO_PEER_COPY)
    if (enable_p2p) {
        ctx->p2p_enabled = ggml_cuda_set_peer_access(device);
    }
#endif

    return cuda_backend;
}

GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

GGML_CALL int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    snprintf(description, description_size, "%s", prop.name);
}

GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));
}

GGML_CALL int ggml_backend_cuda_get_device_cc(int device) {
    if (device < 0 || device >= ggml_cuda_info().device_count) {
        return -1;
    }
    return ggml_cuda_info().devices[device].cc;
}

GGML_CALL const char * ggml_backend_cuda_get_device_pxa_path(int device) {
    if (device < 0 || device >= ggml_cuda_info().device_count) {
        return "";
    }
    return pxa_enhance_path_name(ggml_cuda_info().devices[device].cc);
}

// ---------------------------------------------------------------------------------------------
// PXA per-card serve-flag defaults (2026-09-03).
//
// -b and -ub are not kernel levers, so no resolver in pxa-enhance.cuh can set them: they are
// context parameters chosen before the context is created. But the measured best value for them
// is a property of the CARD SET, and until now every operator had to know the table by heart --
// the campaign chart numbers are all at the values below, and a P100 pair served at the generic
// ub-2048 default gives up a third of its prefill.
//
// The table is exactly the measured cells and nothing more (the measurement ledger,
// entries 2026-09-03 19:00-23:00, Qwable-27B-PXQ4core / PXQ2 on the 1080 Ti):
//
//   2x sm_70   -b 8192 -ub 2048   prefill 1340 t/s @3k, 1282 @20k, decode 39.7 t/s
//   2x sm_60   -b 8192 -ub  256   prefill  340 t/s @3k,  317 @20k, decode 17.8 t/s
//   1x sm_61   -b 2048 -ub  768   prefill 1306 t/s cold (fa-off), 729 chat, decode 59.2 t/s
//   4x sm_60   -b 2048 -ub 2048   Flash-Next 177B hybrid MoE seat, quiet 2026-09-04: prefill 478.27 t/s
//                                 @3121, 400.44 @20801 (BRAG-BOARD-2026-09-05.md:75-76, docs/COOKBOOK.md
//                                 181-223). Added 2026-09-06 (AUTO-DEFAULTS-AUDIT gap b, order #1400a):
//                                 a stock 4x P100 boot fell through to the VRAM ladder, whose uniform-
//                                 split estimate is wrong for the seat's deliberately uneven -ts and
//                                 floored -ub to 512 -> 377.67 t/s @3121 (Phase P alex-ref, #1317).
//                                 -wgt has no auto path; the seat still passes -wgt 8 by hand.
//
// Every other shape returns 0 and keeps whatever the caller had. The mixed V100+P100 cell keeps the
// stock value (its measured lever is -ts, handled separately).
//
// The caller must only apply this to flags the user left unset. REFERENCE stands it down and so
// does PXA_ENHANCE=0, so the level-1 rollback really is the old behaviour end to end.
GGML_CALL int ggml_backend_cuda_pxa_suggest_batch(int * n_batch, int * n_ubatch, const char ** why) {
    if (ggml_pxa_config_level() < 2) {
        return 0;
    }
    const ggml_cuda_device_info & info = ggml_cuda_info();
    const int ndev = info.device_count;
    if (ndev <= 0) {
        return 0;
    }
    int cc0 = info.devices[0].cc;
    for (int i = 1; i < ndev; ++i) {
        if (info.devices[i].cc != cc0) {
            return 0;   // mixed set: no measured cell
        }
    }
    int b = 0, ub = 0;
    const char * name = nullptr;
    if (ndev == 2 && cc0 == 700) {
        b = 8192; ub = 2048; name = "2x V100 (sm_70)";
    } else if (ndev == 2 && cc0 == 600) {
        b = 8192; ub =  256; name = "2x P100 (sm_60)";
    } else if (ndev == 1 && cc0 == 610) {
        b = 2048; ub =  768; name = "1x 1080 Ti (sm_61)";
    } else if (ndev == 4 && cc0 == 600) {
        b = 2048; ub = 2048; name = "4x P100 (sm_60)";
    } else {
        return 0;
    }
    if (n_batch)  *n_batch  = b;
    if (n_ubatch) *n_ubatch = ub;
    if (why)      *why      = name;
    return 1;
}

// ---------------------------------------------------------------------------------------------
// Offline PXQ slab dequant for llama-pxq-export (2026-09-01).
//
// The PXQ types are 64-row panel-interleaved CUDA-consumer formats: their ggml type_traits
// to_float/vec_dot are NULL on purpose, so llama-quantize refuses to requantize from them
// ("cannot requantize from a PXQ slab type"). The un-lock-in path is to decode a PXQ file back
// to F16 first, and the only decoder that is the exact inverse of the quantizer's writer is the
// one the runtime already uses -- ggml_get_to_fp16_cuda(). This entry point exposes it to an
// offline tool without dragging in a backend, a scheduler or a graph.
//
// It is deliberately host->host and stateless: the caller streams panel chunks (a 40-200 GB
// model never fits anywhere), and each call is self-contained so the peak device footprint is
// one chunk in + one chunk out, not one tensor.
GGML_CALL bool pxa_pxq_dequant_host(int device, enum ggml_type src_type, enum ggml_type dst_type,
                                    const void * src, size_t src_bytes,
                                    int64_t nrows, int64_t n_per_row, void * dst) {
    if (src == nullptr || dst == nullptr || nrows <= 0 || n_per_row <= 0) {
        return false;
    }
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        return false;
    }
    // every PXQ kernel hard-aborts on a non-slab-aligned shape -- refuse here instead
    if (nrows % 64 != 0 || n_per_row % 32 != 0) {
        return false;
    }

    to_fp16_cuda_t to_fp16 = nullptr;
    to_fp32_cuda_t to_fp32 = nullptr;
    size_t dst_elem = 0;
    if (dst_type == GGML_TYPE_F16) {
        to_fp16 = ggml_get_to_fp16_cuda(src_type);
        if (to_fp16 == nullptr) return false;
        dst_elem = sizeof(half);
    } else if (dst_type == GGML_TYPE_F32) {
        to_fp32 = ggml_get_to_fp32_cuda(src_type);
        if (to_fp32 == nullptr) return false;
        dst_elem = sizeof(float);
    } else {
        return false;
    }

    const size_t dst_bytes = (size_t) nrows * (size_t) n_per_row * dst_elem;

    ggml_cuda_set_device(device);

    void * d_src = nullptr;
    void * d_dst = nullptr;
    bool ok = false;
    if (cudaMalloc(&d_src, src_bytes) != cudaSuccess) {
        cudaGetLastError();
        goto cleanup;
    }
    if (cudaMalloc(&d_dst, dst_bytes) != cudaSuccess) {
        cudaGetLastError();
        goto cleanup;
    }
    if (cudaMemcpy(d_src, src, src_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaGetLastError();
        goto cleanup;
    }
    // stream 0: the tool is single-threaded and synchronizes below, so the legacy default
    // stream costs nothing and keeps this independent of the backend's stream pool.
    if (to_fp16) {
        to_fp16(d_src, (half *) d_dst, nrows, n_per_row, 0);
    } else {
        to_fp32(d_src, (float *) d_dst, nrows, n_per_row, 0);
    }
    if (cudaStreamSynchronize(0) != cudaSuccess) {
        cudaGetLastError();
        goto cleanup;
    }
    if (cudaMemcpy(dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaGetLastError();
        goto cleanup;
    }
    ok = true;

cleanup:
    if (d_src) cudaFree(d_src);
    if (d_dst) cudaFree(d_dst);
    return ok;
}

// Read-only P2P topology probe: true iff every ordered device pair can peer-access.
// Uses cudaDeviceCanAccessPeer ONLY (no cudaDeviceEnablePeerAccess) so it is side-effect free
// and safe to call at model-load time. Result is cached in a tri-state static.
GGML_CALL bool ggml_backend_cuda_all_pairs_can_peer(void) {
    static int cached = -1; // -1 unknown, 0 false, 1 true
    if (cached != -1) {
        return cached != 0;
    }
    const int device_count = ggml_backend_cuda_get_device_count();
    if (device_count <= 1) {
        cached = 1;
        return true;
    }
    bool all_peer = true;
    for (int id = 0; id < device_count && all_peer; ++id) {
        ggml_cuda_set_device(id);
        for (int id_other = 0; id_other < device_count; ++id_other) {
            if (id == id_other) {
                continue;
            }
            int can_access_peer = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
            if (!can_access_peer) {
                all_peer = false;
                break;
            }
        }
    }
    cached = all_peer ? 1 : 0;
    return all_peer;
}

GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11100 || defined(GGML_USE_MUSA)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();

        GGML_CUDA_LOG_WARN("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    return false;
#endif
}

GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();
    }
}

// backend registry
GGML_CALL static ggml_backend_t ggml_backend_reg_cuda_init(const char * params, void * user_data) {
    ggml_backend_t cuda_backend = ggml_backend_cuda_init((int) (intptr_t) user_data, nullptr);
    return cuda_backend;

    GGML_UNUSED(params);
}

extern "C" GGML_CALL int ggml_backend_cuda_reg_devices();

GGML_CALL int ggml_backend_cuda_reg_devices() {
    int device_count = ggml_backend_cuda_get_device_count();
    //int device_count = 1; // DEBUG: some tools require delaying CUDA initialization
    for (int i = 0; i < device_count; i++) {
        char name[128];
        snprintf(name, sizeof(name), "%s%d", GGML_CUDA_NAME, i);
        ggml_backend_register(name, ggml_backend_reg_cuda_init, ggml_backend_cuda_buffer_type(i), (void *) (intptr_t) i);
    }
    return device_count;
}
