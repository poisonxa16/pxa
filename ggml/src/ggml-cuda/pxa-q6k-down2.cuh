#pragma once
#include "mmvq-verify2.cuh"
bool pxa_q6k_down2_try(int cc,const ggml_tensor *weights,const ggml_tensor *input,
    const ggml_tensor *ids,const ggml_tensor *output,const pxa_mv2_args &args,cudaStream_t stream);
