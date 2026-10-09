#!/bin/bash
#SBATCH --account=phd-acharles42
#SBATCH --partition=batch-impulse
#SBATCH --job-name=dapp-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-cpu-%j.out

# CPU-only steps (batch-impulse starts immediately; never run these on the login node --
# it OOM-kills anything that loads a dataset). Submit from inside this folder:
#
#   sbatch SC_cpu.sh build              # data/raw/DAppSCAN -> data/dappscan_v1.parquet + DATA_CARD.md
#   sbatch SC_cpu.sh baseline           # file-length-only floor -> outputs/length_baseline/
#   sbatch SC_cpu.sh dryrun <model>     # download tokenizer + window every split, no model, no GPU
#   sbatch SC_cpu.sh moe                # MoE gate over finished runs (configs/moe.yaml)
#   sbatch SC_cpu.sh collect            # outputs/RESULTS.md
#   v2 (see submit_v2.sh, which submits these for you):
#   sbatch SC_cpu.sh tfidf --test-fold 0 --val-fold 1 --output-dir outputs/v2/tfidf_f0
#   sbatch SC_cpu.sh blend --runs outputs/v2/modernbert_f0 outputs/v2/tfidf_f0 --output-dir outputs/v2/modernbert_tfidf_f0
#   sbatch SC_cpu.sh cvsummary          # outputs/v2/RESULTS_v2.md
#   v3 (see submit_v3.sh): the same steps with --parquet-path data/v3/dappscan_v3.parquet, outputs under outputs/v3;
#   sbatch SC_cpu.sh cvsummary --outputs ./outputs/v3   # outputs/v3/RESULTS_v3.md
#   sbatch SC_cpu.sh balance --in data/v3/dappscan_v3_all.parquet --out data/v3/dappscan_v3.parquet

set -e
SCVD_ROOT="${SCVD_ROOT:-/work/projects/phd-acharles42/acharles42}"
export SCVD_ROOT
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
mkdir -p logs

STEP="${1:?usage: sbatch SC_cpu.sh build|balance|baseline|dryrun <model>|moe|collect|tfidf|blend|cvsummary}"
shift

use_env() {   # $1 = backend
    if [ "$1" = "openmythos" ]; then
        export PATH="$SCVD_ROOT/miniconda3/envs/openmythos/bin:$PATH"
        source "$SCVD_ROOT/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
        conda activate openmythos 2>/dev/null || true
    else
        spack load py-attrs || true
        source "$SCVD_ROOT/SCenv/bin/activate"
    fi
    source ~/.hf_auth 2>/dev/null || true
}

case "$STEP" in
    build)
        use_env scenv
        python -u -m scvd_dapp build --raw data/raw/DAppSCAN --out data/dappscan_v1.parquet "$@"
        ;;
    baseline)
        use_env scenv
        python -u -m scvd_dapp baseline "$@"
        ;;
    dryrun)
        MODEL="${1:?usage: sbatch SC_cpu.sh dryrun <model>}"; shift
        CFG="configs/${MODEL}.yaml"
        use_env "$(tr -d '\r' < "$CFG" | awk -F: '/^backend:/ {gsub(/[ \t]/, "", $2); print $2}')"
        python -u -m scvd_dapp train --config "$CFG" --dry-run --run-name "dryrun_${MODEL}" "$@"
        ;;
    moe)
        use_env scenv
        python -u -m scvd_dapp moe --config configs/moe.yaml "$@"
        ;;
    collect)
        use_env scenv
        python -u -m scvd_dapp collect "$@"
        ;;
    tfidf)
        use_env scenv
        python -u -m scvd_dapp tfidf "$@"
        ;;
    blend)
        use_env scenv
        python -u -m scvd_dapp blend "$@"
        ;;
    balance)
        use_env scenv
        python -u -m scvd_dapp balance "$@"
        ;;
    cvsummary)
        use_env scenv
        python -u -m scvd_dapp cvsummary "$@"
        ;;
    *)
        echo "unknown step: $STEP" >&2; exit 1 ;;
esac
