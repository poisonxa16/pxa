// Copyright (c) 2026 PXA Network.
#pragma once
#include "../common.cuh"

// PXA_FA_LONGPREFILL (2026-10-15; opt-in, default OFF): sm_60 head-256 prompt attention with
// GQA-packed 32-row blocks. Returns true when it took the node; false = the caller's route runs unchanged.
bool ggml_cuda_pxa_fa_longpf_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
