#!/bin/bash
# v2 retrain (file-level training + TF-IDF blend + 5-fold CV). Run on the login node from this folder:
#
#   bash submit_v2.sh pilot     # fold 0: ModernBERT + SecureBERT 2.0, TF-IDF, length floor, blends (~1 h)
#   bash submit_v2.sh full      # folds 0-4: all five models, TF-IDF, length floor, blends
#   sbatch SC_cpu.sh cvsummary  # when everything is done -> outputs/v2/RESULTS_v2.md
#
# Safe to re-run (the OnDemand terminal can replay commands): it skips any job whose name is
# already queued and any run whose outputs/v2/<run>/test_results.json exists.
# Fold k: test = fold k, validation = fold k+1 (mod 5), train = the other three.
set -eu
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p logs outputs/v2

MODE="${1:-}"
case "$MODE" in
    pilot) FOLDS="0";         MODELS="modernbert securebert2" ;;
    full)  FOLDS="0 1 2 3 4"; MODELS="modernbert securebert2 qwen3 opencoder codellama" ;;
    *) echo "usage: bash submit_v2.sh pilot|full" >&2; exit 1 ;;
esac

mem_for() {
    case "$1" in
        codellama) echo 64G ;;
        qwen3|opencoder) echo 40G ;;
        *) echo 32G ;;
    esac
}
queued() { squeue -u "$USER" -h -o %j | grep -x "$1" > /dev/null; }
finished() { [ -f "outputs/v2/$1/test_results.json" ]; }
jobid() { squeue -u "$USER" -h -o "%i %j" | awk -v n="$1" '$2 == n && !seen {print $1; seen = 1}'; }

for f in $FOLDS; do
    v=$(( (f + 1) % 5 ))
    # CPU: length-only floor and TF-IDF for this fold
    if ! finished "length_f$f" && ! queued "v2-length-f$f"; then
        sbatch --job-name="v2-length-f$f" SC_cpu.sh baseline --test-fold "$f" --val-fold "$v" \
               --output-dir "outputs/v2/length_f$f"
    fi
    if ! finished "tfidf_f$f" && ! queued "v2-tfidf-f$f"; then
        sbatch --job-name="v2-tfidf-f$f" SC_cpu.sh tfidf --test-fold "$f" --val-fold "$v" \
               --output-dir "outputs/v2/tfidf_f$f"
    fi
    for m in $MODELS; do
        # GPU: the model on this fold (memory sized to the model, not SC.sh's 120G default)
        if ! finished "${m}_f$f" && ! queued "v2-$m-f$f"; then
            sbatch --job-name="v2-$m-f$f" --mem="$(mem_for "$m")" SC.sh "${m}_v2" --test-fold "$f" --val-fold "$v" \
                   --run-name "${m}_v2_f$f" --output-dir "outputs/v2/${m}_f$f"
        fi
        # CPU: blend with TF-IDF once both have finished
        if ! finished "${m}_tfidf_f$f" && ! queued "v2-blend-$m-f$f"; then
            deps=""
            for d in "v2-$m-f$f" "v2-tfidf-f$f"; do
                j="$(jobid "$d")"
                if [ -n "$j" ]; then deps="$deps:$j"; fi
            done
            dep=""
            if [ -n "$deps" ]; then dep="--dependency=afterok$deps"; fi
            # shellcheck disable=SC2086
            sbatch --job-name="v2-blend-$m-f$f" $dep SC_cpu.sh blend \
                   --runs "outputs/v2/${m}_f$f" "outputs/v2/tfidf_f$f" --output-dir "outputs/v2/${m}_tfidf_f$f"
        fi
    done
done
squeue -u "$USER" -o '%.9i %.22j %.10P %.3t %.10M %.7m %R'
