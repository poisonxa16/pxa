// Explicit instantiations for the vendored Volta MMA flash-attention kernel at
// head size 512/512, GQA packing 8 (PXA_FA_D512_VOLTA).
//
// The vendored header declares every case `extern template`, so each shape the
// dispatcher can reach has to be instantiated somewhere. One TU per
// (head size, ncols2), as for the 128 and 256 families.
//
// ncols1 rungs: on sm_70 the kernel is compiled out below 32 columns per block
// (fattn-mma-ml.cuh: `if (ncols1*ncols2 < 32) NO_DEVICE_CODE`), so the 16/ncols2
// rung that the 128/256 ladders start from is a STUB here and must never be
// dispatched. The 512 ladder therefore starts at ncols1 = 4 (32 columns) and
// tops out at ncols1 = 8 (64 columns) -- the same two upper rungs the vendored
// Volta ladder uses, with the unusable bottom rung removed.

#include "volta-mma/fattn-mma-ml.cuh"

namespace pxa_volta_fa {
DECL_FATTN_MMA_F16_CASE(512, 512, 4, 8);
DECL_FATTN_MMA_F16_CASE(512, 512, 8, 8);
}
