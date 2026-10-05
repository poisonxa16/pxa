#pragma once

// PXQN revision 1 on the llama side (ggml-pxqn.h is the format): the GGUF keys, the loader's
// revision gate, the site of every weight tensor, and the K granularity a rotated tensor must
// be cut at. One place, so the loader, the graph, the slicer and the encoder cannot disagree
// about which tensors a rotated site owns.

#include "ggml.h"
#include "ggml-pxqn.h"
#include "llama-arch.h"

#include <cstdint>
#include <string>

#define LLAMA_PXQN_KEY_REV        "pxa.pxqn.rev"        // u32, 1 (absent = an old PXQ file)
#define LLAMA_PXQN_KEY_ROT_SEED   "pxa.pxqn.rot.seed"   // u64
#define LLAMA_PXQN_KEY_ROT_SITES  "pxa.pxqn.rot.sites"  // string, comma list of ON sites ("" = none)
#define LLAMA_PXQN_KEY_CALIB      "pxa.pxqn.calib"      // string, sha256 of the calibration corpus
#define LLAMA_PXQN_KEY_ALLOC      "pxa.pxqn.alloc"      // string, hash of the tier map
#define LLAMA_PXQN_KEY_ENCODER    "pxa.pxqn.encoder"    // string, encoder build id

// Reads the pxa.pxqn.* rotation contract. rev = 0 when the file has no pxa.pxqn.rev (an old PXQ
// file, loads as today). THROWS std::runtime_error with a clear message when the revision is
// newer than this build reads (rev > PXQN_REV), when a key has the wrong GGUF type, or when
// rot.sites names a site this build does not know -- each of those would otherwise decode wrong
// without a sound.
void llama_pxqn_read_keys(const struct gguf_context * ctx, uint32_t & rev, uint64_t & seed, uint32_t & sites);

// "attn_in,out_in" <-> OR of PXQN_SITE_BIT(). parse throws on an unknown name.
uint32_t    llama_pxqn_parse_sites(const std::string & s);
std::string llama_pxqn_sites_str(uint32_t sites);
const char * llama_pxqn_site_name(int site);

// The activation site a weight tensor consumes (by GGUF tensor name), PXQN_SITE_NONE when its
// input is not a site (norms, embeddings, the output head, nextn.eh_proj, ...).
int llama_pxqn_tensor_site(const std::string & tensor_name);

// Layer index of a blk.N.* tensor name, -1 otherwise.
int llama_pxqn_tensor_layer(const std::string & tensor_name);

// Which rotation sites this build wires into the graph of architecture `arch` (OR of PXQN_SITE_BIT; 0 = the
// architecture loads PXQN files only UNROTATED). `moe` = the model has routed experts (n_expert > 0): the
// routed-expert FFN sites (ffn_in, down_in) have their own wiring and are listed apart. The loader refuses a file
// whose rot.sites is not a subset of this mask. The Pro encoder carries the same table (pxqe_pro.py ENGINE_ROT_ARCHS,
// keyed by the GGUF architecture name) and falls back to the supported subset for a family not listed, so a file it
// writes always loads: KEEP THE TWO IN LOCKSTEP when a family gains a site.
uint32_t llama_pxqn_arch_rot_sites(llm_arch arch, bool moe);

// Whether this tensor is stored rotated under the given site mask.
static inline bool llama_pxqn_tensor_rotated(const std::string & tensor_name, uint32_t sites) {
    const int site = llama_pxqn_tensor_site(tensor_name);
    return site != PXQN_SITE_NONE && (sites & PXQN_SITE_BIT(site)) != 0;
}
