#!/bin/bash
# Fetch DAppSCAN at the pinned commit -- only the parts the builder reads
# (DAppSCAN-source/contracts + SWCsource + the xlsx: ~200 MB instead of 4.9 GB).
# Run on the LOGIN node (needs internet, light on memory):   bash fetch_dappscan.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

COMMIT=66a56619c44770e05c2db600fa6468115ff0dcd5   # 2025-03-25; the builder was validated at this commit
DEST=data/raw/DAppSCAN

if [ ! -d "$DEST/.git" ]; then
    git clone --filter=blob:none --no-checkout https://github.com/InPlusLab/DAppSCAN.git "$DEST"
fi
cd "$DEST"
git sparse-checkout init --no-cone
git sparse-checkout set '/DAppSCAN-source/contracts/' '/DAppSCAN-source/SWCsource/' \
                        '/Audit_and_Repository_link.xlsx' '/README.md'
git -c advice.detachedHead=false checkout --quiet "$COMMIT"
echo "DAppSCAN checked out at $(git rev-parse HEAD)"
echo "  .sol files:        $(find DAppSCAN-source/contracts -name '*.sol' | wc -l)   (expected 21457)"
echo "  annotation files:  $(find DAppSCAN-source/SWCsource -name '*.json' | wc -l)   (expected 948)"
du -sh . 2>/dev/null | awk '{print "  size on disk:      " $1}'
