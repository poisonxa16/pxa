// Explicit instantiation of the vendored no-tensor-core tile flash-attention
// kernel at head size 512/512 (PXA_FA_D512_VOLTA=2). See volta-tile/ for
// provenance and fattn-volta-tile.cu for the dispatch.

#include "volta-tile/fattn-tile-ml.cuh"

namespace pxa_volta_fa {
DECL_FATTN_TILE_CASE(512, 512);
}
