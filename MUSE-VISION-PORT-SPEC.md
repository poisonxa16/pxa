# MUSE-GLIMMER VISION PORT — worktree spec (mgv-wt @ e53cb091 = build-fug baseline)

Goal: register PROJECTOR_TYPE_MUSE_GLIMMER in OUR examples/mtmd so our llama-server serves vision on
a production PXQ4 instance. Baseline commit e53cb091 (== build-fug) so the text-path byte-identical
gate is meaningful. BUILD INTO A FRESH DIR (build-fugv), NEVER build-fug (serves the live production
instance).

## Status
- [x] enum PROJECTOR_TYPE_MUSE_GLIMMER + name "muse-glimmer" added to clip-impl.h (registration).
- [x] hparams field muse_glimmer_sparse_factor + load branch (n_merge=2, rope 10000, sf=4, limit 1..4096 tok, warmup 32x32)
- [x] tensor loading (mm_0/1/2_w via TN_LLAVA_PROJ; tower is a plain biased ViT - no qk-norms, no ls)
- [x] graph builder build_muse_glimmer() - inline windowed loop in the build_qwen2vl idiom (2D RoPE W|H, sp_mask on sparse layers, global = last OR every 4th, pixel-shuffle via ds_perm, mm0-erfGELU-mm1-erfGELU-mm2)
- [x] set_input branch (VERBATIM transplant of the perm/mask math)
- [x] preprocessor (muse_glimmer_grid_size transplant + stretch resize; bicubic in lieu of Lanczos - our img_tool has no Lanczos)
- [x] dispatch, n_output_tokens x/y/n, clip_n_mmproj_embd (mm_2_w->ne[1]), mtmd.cpp img markers <|image_start|>/<|image_end|>
- [~] build build-fugv IN PROGRESS (cmake configured OK; clip.cpp+mtmd.cpp pass standalone -fsyntax-only); Gate 1 (image read) + Gate 2 (text byte-identical vs build-fug) NEED a 2-card window - coordinator arranges

## ⭐ REUSE MAP (why this is ~80% infra we already have)
Our clip_graph (examples/mtmd/clip.cpp) ALREADY has, and muse needs:
- build_vit with PER-LAYER window masking: lines ~767-881, gated on hparams.n_wa_pattern>0. It builds a
  named "window_mask" f32 input and per layer sets attn_mask = (il+1)%n_wa_pattern==0 ? nullptr : mask.
  MUSE maps sparse_factor=4 -> n_wa_pattern=4. DELTA: muse ALSO forces the LAST layer global
  (il==n_layer-1). Either set n_wa_pattern=4 and additionally null the mask on the last layer, or add a
  muse flag in the build_vit layer loop.
- build_rope_2d (2D RoPE, W then H) — muse add_pos = build_rope_2d(cur,pos_w,pos_h,rope_theta=10000,false)
- build_attn, build_ffn (FFN_GELU_ERF), build_norm (NORM_TYPE_NORMAL) — all present
- build_patch_merge_permute — candidate for the pixel-shuffle downsample; muse uses an explicit ds_perm
  gather instead. Simplest: transplant mainline's ds_perm path (ggml_get_rows + reshape/permute/cont).
- model.mm_0_w/mm_1_w/mm_2_w already in the model struct (the 3-layer adapter: 6144->4096->4096->6656)

NOTE the muse graph groups patches by an explicit sp_perm + block-diagonal sp_mask (computed host-side),
which is a DIFFERENT expression of windowing than qwen25vl's window_index gather. So build the muse graph
either (a) inline (transplant mainline models/muse-glimmer.cpp build() using our helpers) applying
sp_perm/inv_perm around a windowed build_vit, or (b) fold sp_perm into a window_index-style gather. (a) is
the faithful path and matches the set_input below.

## HPARAMS (add to clip_hparams in clip.cpp; RESIZE_ALGO_LANCZOS exists? if not use LANCZOS/BICUBIC)
    int32_t muse_glimmer_patch_temporal = 0;
    int32_t muse_glimmer_sparse_factor  = 0;
Hparams load branch (adapt from mainline clip.cpp ~1579):
    case PROJECTOR_TYPE_MUSE_GLIMMER: {
        hparams.n_merge = 2;
        hparams.image_resize_algo = RESIZE_ALGO_LANCZOS;   // fallback: BICUBIC
        hparams.rope_theta = 10000.0f;
        hparams.muse_glimmer_patch_temporal = 2;
        hparams.muse_glimmer_sparse_factor  = 4;
        hparams.n_wa_pattern = 4;                          // <-- reuse our windowing
        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
        hparams.set_limit_image_tokens(1, 4096);
        hparams.set_warmup_n_tokens(32*32);
    } break;

## GRAPH (reference: mainline tools/mtmd/models/muse-glimmer.cpp build(), 88 lines)
Pipeline: build_inp (conv2d patchify) -> + resize_position_embeddings(BILINEAR) -> get_rows(sp_perm) ->
build_vit(x, n_tok, NORM_TYPE_NORMAL, FFN_GELU_ERF, nullptr, add_pos=build_rope_2d) with per-layer masks
(sparse layers = sp_mask, globals=null; global = last layer OR (il+1)%4==0) -> get_rows(inv_perm) ->
pixel-shuffle via ds_perm: get_rows(ds_perm), reshape [n_embd, f*f, n_out], permute(1,0,2,3), cont,
reshape [n_embd*f*f, n_out] -> mm_0 -> gelu_erf -> mm_1 -> gelu_erf -> mm_2  => [6656, n_out].
Named f32/i32 inputs the graph must create (ggml_set_input): muse_glimmer_{sp_perm,inv_perm,pos_w,pos_h,
ds_perm} (I32,[n_tok]) and muse_glimmer_sp_mask (F32,[n_tok,n_tok]).

## SET_INPUT (VERBATIM transplant — mainline clip.cpp 4221; our set_input_i32/f32 exist for qwen25vl)
[see the block appended below — copy exactly; a wrong permutation loads fine and emits garbage]

## PREPROCESSOR (mainline mtmd-image.cpp 1620) — DRIFT: our mtmd-image API differs
mainline uses mtmd_image_preproc_out / img_tool::resize / output.append. Our examples/mtmd/mtmd-image.*
is the older ik-fork shape — adapt muse_glimmer_grid_size (aspect-preserving, cap = image_max_pixels/
patch_area, PIL stretch resize, no pad) to our preprocess signature. Reference grid fn is self-contained.

## BUILD + GATES
Build: docker run --runtime=nvidia (a FREE 2-card set; NOT 3, NOT a production instance's 2,4 if busy) with the CUDA
image, cmake -B build-fugv -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=70 ... ; cmake --build build-fugv
-j --target llama-server. (mtmd is already linked; --mmproj is live.)
Gate 1: launch build-fugv llama-server -m <base Q4_K_XL or abl PXQ4> --mmproj <mmproj-*.gguf> on a free
pair; feed tools/mtmd/test-1.jpeg -> must describe the NYT-1969 moon-landing page (the proven-good ref).
Gate 2: temp-0 text-only prompt, build-fugv vs build-fug -> outputs BYTE-IDENTICAL (vision must not
perturb the text graph). If not identical, the port touched a shared path — find it before any swap.

## APPENDIX A — set_input branch (transplant verbatim):
```cpp
        case PROJECTOR_TYPE_MUSE_GLIMMER:
            {
                const int grid_w = pos_w;            // image_size_width  / patch_size
                const int grid_h = pos_h;            // image_size_height / patch_size
                const int n_tok  = grid_w * grid_h;
                const int pgrid  = (int) std::sqrt((double) ctx->model.position_embeddings->ne[1]); // 32
                const int f      = hparams.n_merge;  // downsample 2

                // pixel patchify runs inside the graph via build_inp() (ggml_conv_2d);
                // pos-emb bilinear interp via resize_position_embeddings().

                // --- sparse window grouping (pgrid x pgrid windows) ---
                const int win = pgrid;
                const int nwin_h = (grid_h + win - 1) / win;
                const int nwin_w = (grid_w + win - 1) / win;
                std::vector<int32_t> sp_perm; sp_perm.reserve(n_tok);
                std::vector<int>     sp_slens;
                for (int wy = 0; wy < nwin_h; wy++) {
                    for (int wx = 0; wx < nwin_w; wx++) {
                        int cnt = 0;
                        for (int hh = 0; hh < win; hh++) {
                            for (int ww = 0; ww < win; ww++) {
                                const int gy = wy * win + hh;
                                const int gx = wx * win + ww;
                                if (gy < grid_h && gx < grid_w) { sp_perm.push_back(gy * grid_w + gx); cnt++; }
                            }
                        }
                        if (cnt > 0) sp_slens.push_back(cnt);
                    }
                }
                std::vector<int32_t> rpos_w(n_tok), rpos_h(n_tok), inv_perm(n_tok);
                for (int i = 0; i < n_tok; i++) {
                    const int orig = sp_perm[i];
                    rpos_w[i] = (orig % grid_w) + 1; // 1-indexed
                    rpos_h[i] = (orig / grid_w) + 1;
                    inv_perm[orig] = i;
                }
                set_input_i32("muse_glimmer_sp_perm",  sp_perm);
                set_input_i32("muse_glimmer_inv_perm", inv_perm);
                set_input_i32("muse_glimmer_pos_w",    rpos_w);
                set_input_i32("muse_glimmer_pos_h",    rpos_h);

                // block-diagonal window mask (permuted order)
                std::vector<float> sp_mask((size_t) n_tok * n_tok, -INFINITY);
                {
                    int off = 0;
                    for (int s : sp_slens) {
                        for (int a = 0; a < s; a++)
                            for (int b = 0; b < s; b++)
                                sp_mask[(size_t) (off + a) * n_tok + (off + b)] = 0.0f;
                        off += s;
                    }
                }
                set_input_f32("muse_glimmer_sp_mask", sp_mask);

                // pixel-shuffle gather (original order): f*f spatial neighbours grouped
                std::vector<int32_t> dsp; dsp.reserve(n_tok);
                for (int oy = 0; oy < grid_h / f; oy++)
                    for (int ox = 0; ox < grid_w / f; ox++)
                        for (int ry = 0; ry < f; ry++)
                            for (int rx = 0; rx < f; rx++)
                                dsp.push_back((oy * f + ry) * grid_w + (ox * f + rx));
                set_input_i32("muse_glimmer_ds_perm", dsp);
            } break;
        case PROJECTOR_TYPE_MINICPMV:
            {
                // inspired from siglip:
                //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit
```

## APPENDIX B — graph build() reference:
```cpp
#include "models.h"

// MuseGlimmer vision encoder: 50-layer ViT with 2D RoPE, sparse block-diagonal
// window attention (every 4th + last layer global), pixel-shuffle downsample, then
// adapter MLP + LLM's vision_projection.
//
// Several quantities are precomputed on host and fed as named graph inputs (filled in
// clip.cpp set_input, PROJECTOR_TYPE_MUSE_GLIMMER branch):
//   muse_glimmer_pos_w/_h [n_tok] i32         : 1-indexed RoPE positions (sparse-permuted order)
//   muse_glimmer_sp_perm  [n_tok] i32         : window grouping permutation (applied after ln_pre)
//   muse_glimmer_inv_perm [n_tok] i32         : inverse of sp_perm (applied after blocks)
//   muse_glimmer_ds_perm  [n_tok] i32         : pixel-shuffle gather (original order)
//   muse_glimmer_sp_mask  [n_tok, n_tok] f32  : block-diagonal window mask (sparse layers)
ggml_cgraph * clip_graph_muse_glimmer::build() {
    const int ds = hparams.n_merge;              // downsample factor (2)
    const int sf = hparams.muse_glimmer_sparse_factor;   // 4
    const int n_tok     = n_patches;
    const int n_out     = (n_patches_x / ds) * (n_patches_y / ds);
    const float rope_base = hparams.rope_theta;  // 10000

    auto inp_i32 = [&](const char * name, int64_t n) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n);
        ggml_set_name(t, name);
        ggml_set_input(t);
        return t;
    };

    ggml_tensor * pos_w    = inp_i32("muse_glimmer_pos_w",    n_tok);
    ggml_tensor * pos_h    = inp_i32("muse_glimmer_pos_h",    n_tok);
    ggml_tensor * sp_perm  = inp_i32("muse_glimmer_sp_perm",  n_tok);
    ggml_tensor * inv_perm = inp_i32("muse_glimmer_inv_perm", n_tok);
    ggml_tensor * ds_perm  = inp_i32("muse_glimmer_ds_perm",  n_tok);

    ggml_tensor * sp_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tok, n_tok);
    ggml_set_name(sp_mask, "muse_glimmer_sp_mask");
    ggml_set_input(sp_mask);

    // patchify via build_inp (conv2d over raw pixels) + bilinear-resized learned pos-emb
    ggml_tensor * x = build_inp();                                                     // [n_embd, n_tok, 1]
    x = ggml_add(ctx0, x, resize_position_embeddings(GGML_SCALE_MODE_BILINEAR));
    cb(x, "after_posemb", -1);

    // group patches into pgrid x pgrid windows (sparse attention order)
    x = ggml_get_rows(ctx0, x, sp_perm);
    cb(x, "after_sp_perm", -1);

    // per-layer mask: sparse layers get sp_mask, global layers (every sf-th and last) get none
    std::vector<ggml_tensor *> attn_mask_layers(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        const bool is_global = (il == n_layer - 1) || ((il + 1) % sf == 0);
        attn_mask_layers[il] = is_global ? nullptr : sp_mask;
    }

    // 2D RoPE: first half of head_dim uses width pos, second half uses height pos
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return build_rope_2d(ctx0, cur, pos_w, pos_h, rope_base, false);
    };

    build_vit_opts opts;
    opts.attn_mask_layers = std::move(attn_mask_layers);

    // pre_ln, per-layer transformer, post_ln (all inside build_vit); reference uses exact (erf) GELU
    x = build_vit(x, n_tok, NORM_TYPE_NORMAL, FFN_GELU_ERF, nullptr, add_pos, opts);

    // un-permute back to original grid order
    x = ggml_get_rows(ctx0, x, inv_perm);
    cb(x, "after_inv_perm", -1);

    // pixel-shuffle downsample: gather f*f spatial neighbors then concat channel-outer.
    // out[c*(ds*ds)+s, o] = x[ds_perm gathered][o*(ds*ds)+s, c]
    x = ggml_get_rows(ctx0, x, ds_perm);                 // [n_embd, n_tok], grouped
    x = ggml_reshape_3d(ctx0, x, n_embd, ds * ds, n_out);// [c, s, o]
    x = ggml_permute(ctx0, x, 1, 0, 2, 3);               // [s, c, o]
    x = ggml_cont(ctx0, x);
    x = ggml_reshape_2d(ctx0, x, n_embd * ds * ds, n_out); // [6144, n_out]
    cb(x, "encoder_out", -1);

    // adapter (6144->4096->4096, exact GELU each) + LLM vision_projection (4096->6656)
    x = build_mm(model.mm_0_w, x);
    x = ggml_gelu_erf(ctx0, x);
    x = build_mm(model.mm_1_w, x);
    x = ggml_gelu_erf(ctx0, x);
    x = build_mm(model.mm_2_w, x);                       // [6656, n_out]
    cb(x, "projected", -1);

    ggml_build_forward_expand(gf, x);
    return gf;
}
```
