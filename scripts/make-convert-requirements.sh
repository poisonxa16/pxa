#!/bin/sh
# make-convert-requirements.sh <requirements dir>   -> stdout: tools/requirements-convert.txt
#
# The python packages tools/convert_hf_to_gguf.py (Hugging Face model -> GGUF, run by PXA Control's Encode tab) imports. One
# renderer for the release tarball (scripts/make-release-tarball.sh) and the container image (docker/Dockerfile), so the two lists
# cannot drift. Taken from the repo's own requirement files at the packaged commit, minus the "gguf" line: the converter uses the
# gguf python package that travels with it (gguf-py/, put first on sys.path), so a pip "gguf" is never needed and never shadows it.
# safetensors is added because the Encode tab's own tool check names it.
set -e
D=${1:?usage: make-convert-requirements.sh <requirements dir>}
echo "# Python packages for convert_hf_to_gguf.py (Hugging Face model -> GGUF), the converter PXA Control's Encode tab runs."
echo "# pip install -r requirements-convert.txt     (into a virtual environment; set PXA_CONVERT_PYTHON to its python)"
echo "# The gguf package is NOT listed: the gguf-py/ next to the converter is the one of this engine commit and is used first."
echo "# Taken from requirements-convert_hf_to_gguf.txt and requirements-convert_legacy_llama.txt of this commit."
{ cat "$D/requirements-convert_hf_to_gguf.txt" "$D/requirements-convert_legacy_llama.txt" \
    | grep -v -E '^[[:space:]]*(-r[[:space:]]|#|$)' | grep -v -i -E '^gguf([<>=~ ]|$)'
  echo "safetensors"; } | awk '!seen[$0]++'
