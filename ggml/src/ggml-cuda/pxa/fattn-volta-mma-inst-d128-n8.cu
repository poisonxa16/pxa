// Explicit instantiations for the vendored mainline Volta MMA flash-attention
// kernel (see volta-mma/ for provenance and fattn-volta-mma.cu for the dispatch).
// The vendored header declares every case `extern template`, so each shape this
// fork can dispatch has to be instantiated somewhere. Split one TU per
// (head size, ncols2) so they compile in parallel, as mainline does.
//
// ncols1 rungs are mainline's Volta ladder (fattn.cu:145-166 with
// turing_mma_available() false): 16/ncols2, 32/ncols2, 64/ncols2.

#include "volta-mma/fattn-mma-ml.cuh"

namespace pxa_volta_fa {
DECL_FATTN_MMA_F16_CASE(128, 128, 16/8, 8);
DECL_FATTN_MMA_F16_CASE(128, 128, 32/8, 8);
DECL_FATTN_MMA_F16_CASE(128, 128, 64/8, 8);
}
