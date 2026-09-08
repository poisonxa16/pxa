# Box paths for the pxq4 harnesses

The tracked harness code names no machine. Paths come from environment variables (generic defaults
in parentheses); a per-box `pxa/pxq4/box.env` (gitignored) sets them, and every shell harness sources
it when present. Lane-only scripts that hardcode one box are `export-ignore` in `.gitattributes` and
never reach a release archive.

| variable | meaning | default |
|---|---|---|
| `PXA_MODELS_COLD` | archival model store (GGUFs) | `./models` |
| `PXA_MODELS` | served checkpoints (vLLM safetensors) | `./models` |
| `PXA_MODELS_HOT` | fast-tier checkpoints | `./models` |
| `PXA_PXQ4_PKG` | the `pxa/pxq4` package root (kernels, sidecar) | `.` |
| `PXA_PASCALOPS_ROOT`, `PXA_PASCALOPS_RESULTS` | Pascal-ops source tree and its results dir | `.`, `./results` |
| `PXA_PP_ROOT`, `PXA_PP_RESULTS` | vLLM pipeline-parallel source tree and its results dir | `.`, `./results` |
| `PXA_VLLM_SRC` | the vLLM fork source tree | `/opt/vllm` |
| `PXA_ENGINE_WT` | the llama engine tree (reference server) | `.` |
| `PXA_CAMPAIGN` | the speed-campaign directory (prompts, logs) | `./speed-campaign` |
| `PXA_VLLM_PLUGIN` | native encoder / plugin libs | `./plugin` |
| `PXA_CACHEONE`, `PXA_CACHETWO`, `PXA_USER` | raw mount roots (only where a script mounts a whole pool) | `/tmp` |
| `PXA_BOX_HOSTNAME` | when set, scripts refuse to run on any other host | unset |

`python -m gguf_to_vllm.boxenv` prints the resolved values; harness Python reads `os.environ` directly.
