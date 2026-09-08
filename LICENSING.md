# Licensing map

This repository is one clone containing two engines under two different
permissive licences. They are compatible, but they are not the same licence,
and each keeps its own notices.

| path | what | licence | upstream |
|---|---|---|---|
| repository root | `pxa` inference engine (C/C++) | **MIT** — see `LICENSE` | llama.cpp -> ik_llama.cpp |
| `tools/vllm-pxq4/`, `pxa/pxq4/` | the PXQ4 quantization plugin for vLLM (Python + CUDA) | **Apache-2.0**, matching vLLM — see `tools/vllm-pxq4/LICENSE-NOTICE.md` | vLLM -> 1Cat-vLLM |

Attribution for each side lives with that side: `NOTICE` at the root for the engine,
`tools/vllm-pxq4/LICENSE-NOTICE.md` for the vLLM plugin. Apache-2.0 section 4 requires that
notice be carried into redistributions; do not drop it. The plugin patches no vLLM source — it
registers through the documented `register_quantization_config` hook — but it is Apache-2.0
because that is vLLM's licence and the plugin is derived work against its interfaces.

The vLLM *serving stack itself* is not vendored into this repository; run it from upstream vLLM
or from 1Cat-vLLM (which carries the sm_70 support this plugin depends on) and install the
plugin alongside it. `docs/PXA-SM70-SERVING.md` and `docs/VLLM.md` are the build and serving
instructions.

The upstream engine README is preserved rather than deleted:
`docs/README-upstream-ik_llama.md`.

The engine sits at the repository root rather than under `engine/` because it
is the original tree and moving it would break every documented path, every
build recipe and every issue link written to date. The split is by licence,
not by symmetry of directory names.
