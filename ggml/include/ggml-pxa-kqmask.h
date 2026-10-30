// ggml-pxa-kqmask.h -- PXA_KV_BAND_ANCHOR0: the layout of the proof llama.cpp writes
// about a KQ mask it just filled into the mask tensor's op_params, right before a compute (cleared after it); the
// scheduler (ggml-backend.cpp) copies an input's proof onto its device copy, and a backend may then skip the KV-band
// anchor work for a soft_max / P.V whose row and column counts match the proof (ggml-cuda/pxa/kv-band.cu). All zero =
// no proof. Its own header so the few files that use it are the only ones a change rebuilds.
#pragma once

#define GGML_PXA_KQMASK_PROOF_MAGIC 0x50584130 // slot 0
#define GGML_PXA_KQMASK_PROOF_ROWS  1          // rows scanned (the ubatch tokens), every plane
#define GGML_PXA_KQMASK_PROOF_COLS  2          // columns scanned (= ne[0])
#define GGML_PXA_KQMASK_PROOF_COL0  3          // 1 = column 0 is open (not -inf) in every scanned row
#define GGML_PXA_KQMASK_PROOF_LAST  4          // the largest open column over the scanned rows, -1 none, -2 not computed
#define GGML_PXA_KQMASK_PROOF_FIRST 5          // the smallest open column over the scanned rows, -1 none, -2 not computed
                                               // (PXA_KV_BAND_HOSTPROOF; written with LAST for decode-sized ubatches, and
                                               // then also when column 0 is closed: COL0 = 0)
#define GGML_PXA_KQMASK_PROOF_END   6          // slots [0, END) are the proof

// PXA_KV_BAND_VPAD: llama.cpp places a cleared pad of V's own type right behind an f16 V
// cache tensor and marks the V tensor's op_params with how many bytes past its end stay inside that pad (same buffer,
// zero, never written), so the Pascal KV band may read a matrix moved by its anchor in place past the tensor end
// (ggml-cuda/pxa/kv-band.cu, pxa_kv_band_limit). All zero = no pad.
#define GGML_PXA_KV_VPAD_MAGIC      0x50584156 // slot 0 of the V cache tensor's op_params
#define GGML_PXA_KV_VPAD_BYTES      1          // readable zero bytes behind the tensor end
