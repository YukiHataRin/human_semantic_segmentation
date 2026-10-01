#!/bin/zsh
set -eu
cd "${0:A:h}"
export PYTHONNOUSERSITE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec /opt/miniconda3/bin/conda run --no-capture-output -p ./.conda python -m studio "$@"
