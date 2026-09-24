#!/bin/bash
# Usage: bash examples/mkm590/sync.sh JOB_ID [--watch]
# Pull lightweight monitoring output. Never transfer the large restart state.
set -euo pipefail

JOB_ID=${1:?Supply the Slurm job number}
WATCH=${2:-}
case "$JOB_ID" in *[!0-9]*|'') echo "JOB_ID must be numeric" >&2; exit 1;; esac
case "$WATCH" in ''|--watch) ;; *) echo "Use --watch for updates every 60 seconds" >&2; exit 1;; esac

HOST=dl1f12@loginx001.iridis.soton.ac.uk
REMOTE=/scratch/dl1f12/channelflow-gpu/code
HERE=$(cd "$(dirname "$0")" && pwd)
LOCAL="$HERE/remote-output"
KEY="$HOME/.ssh/id_ed25519_iridis"
mkdir -p "$LOCAL"

sync_once() {
    ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=15 "$HOST" \
        "sacct -j $JOB_ID --noheader --parsable2 --format=JobIDRaw,State,Elapsed,NodeList" \
        > "$LOCAL/job-status.txt.tmp" || return
    mv "$LOCAL/job-status.txt.tmp" "$LOCAL/job-status.txt"

    # Only named monitoring files are mirrored. --delete removes stale slices
    # after a restart rollback; excluded local plots/animations are protected.
    rsync -az --delete -e "ssh -i $KEY -o BatchMode=yes -o ConnectTimeout=15" \
        --include='config.toml' --include='history.csv' \
        --include='profiles.csv' --include='statistics.toml' \
        --include='slice-*.bin' --exclude='*' \
        "$HOST:$REMOTE/examples/mkm590/output/" "$LOCAL/" || return

    rsync -az -e "ssh -i $KEY -o BatchMode=yes -o ConnectTimeout=15" \
        "$HOST:$REMOTE/channel-mkm590-$JOB_ID.log" "$LOCAL/run.log" || return

    date '+%Y-%m-%d %H:%M:%S %Z' > "$LOCAL/last-sync.txt"

    cat "$LOCAL/last-sync.txt" "$LOCAL/job-status.txt"
    tail -3 "$LOCAL/history.csv"
}

while true; do
    if ! sync_once; then
        echo "Sync failed; the previous local files remain available." >&2
        [ "$WATCH" = --watch ] || exit 1
    fi
    [ "$WATCH" = --watch ] || break
    # Stop polling after the top-level job has ended. Ctrl-C also stops locally.
    if grep -Eq "^$JOB_ID\|(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)" "$LOCAL/job-status.txt"; then
        break
    fi
    sleep 60
done
