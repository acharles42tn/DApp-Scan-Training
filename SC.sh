#!/bin/bash
#SBATCH --account=phd-acharles42
#SBATCH --partition=gpu-warp
#SBATCH --job-name=dapp
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=7-0
#SBATCH --output=logs/slurm-%j.out

# DAppSCAN training -- one model per job. Submit from inside this folder:
#
#   cd $SCVD_ROOT/code/DAppSCAN_Training
#   sbatch SC.sh modernbert                                                   # full run
#   sbatch SC.sh modernbert --max-samples 200 --epochs 1 --run-name smoke_modernbert   # smoke test
#
# The Python environment follows the config's backend:
#   openmythos      -> $SCVD_ROOT/miniconda3/envs/openmythos   (where OpenMythos was built and benchmarked)
#   everything else -> $SCVD_ROOT/SCenv                        (where every Slither model was trained)
# Never call `python -m scvd_dapp train` directly on a GPU node without the right env.

set -e
SCVD_ROOT="${SCVD_ROOT:-/work/projects/phd-acharles42/acharles42}"
export SCVD_ROOT
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
mkdir -p logs

MODEL="${1:?usage: sbatch SC.sh <config name in configs/> [extra train args]}"
shift
CFG="configs/${MODEL}.yaml"
[ -f "$CFG" ] || { echo "ERROR: $CFG not found" >&2; exit 1; }
BACKEND=$(tr -d '\r' < "$CFG" | awk -F: '/^backend:/ {gsub(/[ \t]/, "", $2); print $2}')

if [ "$BACKEND" = "openmythos" ]; then
    CONDA_ENV="$SCVD_ROOT/miniconda3/envs/openmythos"
    export PATH="$CONDA_ENV/bin:$PATH"
    source "$SCVD_ROOT/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate openmythos 2>/dev/null || true
else
    spack load py-attrs || true
    source "$SCVD_ROOT/SCenv/bin/activate"
fi
source ~/.hf_auth 2>/dev/null || true   # exports HF_TOKEN

echo "=== $(date) | job ${SLURM_JOB_ID:-local} | config $CFG | backend $BACKEND ==="
echo "python: $(which python)"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"

python -u -m scvd_dapp train --config "$CFG" "$@"
