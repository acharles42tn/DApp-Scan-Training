#!/bin/bash
# v3 retrain: balanced table (every labelled file + 1 unlabelled file per labelled file, in every fold),
# all 34 SWC types trained (the 15 with >= 20 files scored), stronger oversampling, and each model
# both from its public base weights (<model>_v3) and from its Slither-trained weights (<model>_v3s).
# Run on the login node from this folder ($ROOT/code/DAppSCAN_Training_v3 -- v3's own copy of the package;
# DAppSCAN_Training is used by the mixed-dataset work):
#
#   bash submit_v3.sh pilot                    # fold 0: 5 models x {base, Slither init}, TF-IDF, length floor, blends
#   bash submit_v3.sh full <cfg> [<cfg> ...]   # folds 1-4 for the configs that won on fold-0 VALIDATION,
#                                              # e.g.  bash submit_v3.sh full modernbert_v3s securebert2_v3 ...
#   sbatch SC_cpu.sh cvsummary --outputs ./outputs/v3   # when everything is done -> outputs/v3/RESULTS_v3.md
#
# Safe to re-run: skips any job whose name is queued and any run whose test_results.json exists.
# Fold k: test = fold k, validation = fold k+1 (mod 5), train = the other three.
set -eu
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs outputs/v3
PQ=data/v3/dappscan_v3.parquet
[ -f "$PQ" ] || { echo "missing $PQ" >&2; exit 1; }

MODE="${1:-}"
shift || true
case "$MODE" in
    pilot) FOLDS="0"
           CFGS="modernbert_v3 modernbert_v3s securebert2_v3 securebert2_v3s qwen3_v3 qwen3_v3s opencoder_v3 opencoder_v3s codellama_v3 codellama_v3s" ;;
    full)  FOLDS="1 2 3 4"; CFGS="$*"
           [ -n "$CFGS" ] || { echo "usage: bash submit_v3.sh full <cfg> [<cfg> ...]" >&2; exit 1; } ;;
    *) echo "usage: bash submit_v3.sh pilot | full <cfg> [<cfg> ...]" >&2; exit 1 ;;
esac
for c in $CFGS; do [ -f "configs/$c.yaml" ] || { echo "no configs/$c.yaml" >&2; exit 1; }; done

mem_for() {
    case "$1" in
        codellama*) echo 64G ;;
        qwen3*|opencoder*) echo 40G ;;
        *) echo 32G ;;
    esac
}
queued() { squeue -u "$USER" -h -o %j | grep -x "$1" > /dev/null; }
finished() { [ -f "outputs/v3/$1/test_results.json" ]; }
jobid() { squeue -u "$USER" -h -o "%i %j" | awk -v n="$1" '$2 == n && !seen {print $1; seen = 1}'; }

for f in $FOLDS; do
    v=$(( (f + 1) % 5 ))
    # CPU: length-only floor and TF-IDF on the same balanced files
    if ! finished "length_f$f" && ! queued "v3-length-f$f"; then
        sbatch --job-name="v3-length-f$f" SC_cpu.sh baseline --parquet-path "$PQ" --test-fold "$f" --val-fold "$v" \
               --output-dir "outputs/v3/length_f$f"
    fi
    if ! finished "tfidf_f$f" && ! queued "v3-tfidf-f$f"; then
        sbatch --job-name="v3-tfidf-f$f" SC_cpu.sh tfidf --parquet-path "$PQ" --test-fold "$f" --val-fold "$v" \
               --output-dir "outputs/v3/tfidf_f$f"
    fi
    for c in $CFGS; do
        if ! finished "${c}_f$f" && ! queued "v3-$c-f$f"; then
            sbatch --job-name="v3-$c-f$f" --mem="$(mem_for "$c")" SC.sh "$c" --test-fold "$f" --val-fold "$v" \
                   --run-name "${c}_f$f" --output-dir "outputs/v3/${c}_f$f"
        fi
        if ! finished "${c}_tfidf_f$f" && ! queued "v3-blend-$c-f$f"; then
            deps=""
            for d in "v3-$c-f$f" "v3-tfidf-f$f"; do
                j="$(jobid "$d")"
                if [ -n "$j" ]; then deps="$deps:$j"; fi
            done
            dep=""
            if [ -n "$deps" ]; then dep="--dependency=afterok$deps"; fi
            # shellcheck disable=SC2086
            sbatch --job-name="v3-blend-$c-f$f" $dep SC_cpu.sh blend --parquet-path "$PQ" \
                   --runs "outputs/v3/${c}_f$f" "outputs/v3/tfidf_f$f" --output-dir "outputs/v3/${c}_tfidf_f$f"
        fi
    done
done
squeue -u "$USER" -o '%.9i %.26j %.10P %.3t %.10M %.7m %R'
