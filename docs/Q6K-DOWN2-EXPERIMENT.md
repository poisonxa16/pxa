# Experimental two-token Q6_K down projection

`PXA_Q6K_DOWN2=1` enables a narrow SM70 path at the existing fused MoE down
projection. Default is off. Unsupported layouts, biases, quantization types and
batch widths retain the existing path. Up/gate is unchanged.

The initial supported synthetic geometry is K=512, rows=2048, 256 experts,
eight routed IDs per token and two tokens. IDs must be in range. Repeated IDs
are computed independently; weight sharing is restricted to one-to-one matches. No model artifact is included or selected by this option.

Four independent warps share a block. Where the two tokens select an expert in
common, one occurrence computes both activation rows; the second occurrence
exits. Each output retains the existing Q6_K/Q8_1 dot-product walk and warp
reduction. There is no expanded-weight copy or additional workspace.

## Synthetic test

Configure a CUDA build with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70`
and `-DPXA_BUILD_Q6K_DOWN2_TEST=ON`, then build `test-cuda-q6k-down2`.
Run `build/bin/test-cuda-q6k-down2 0` on a suitable device. The test creates
seeded synthetic weights/activations; it requires no model, prompt or network.
It compares every output and padding sentinel with the per-token reference,
checks selected outputs against an independent scalar calculation, and measures
all overlap counts from zero through eight with compact and padded strides,
plus repeated IDs in either token.
`PXA_DOWN2_QUICK=1` selects a reduced correctness run for memory checking.

Run Compute Sanitizer memcheck against the same executable before qualification.
A full backend test run and integrated serving comparison remain required.
This draft has not yet been built or GPU-tested against the current upstream
revision. No end-to-end performance or model-quality result is claimed here.
