#!/usr/bin/env bash
# Development build and run from this working tree (no installer, no auto-update).
#
#   tools/dev.sh                 run the last build (builds first only if there is none)
#   tools/dev.sh build           (re)build
#   tools/dev.sh build-run       (re)build, then run
#   tools/dev.sh run -- args...  run with game arguments
#
# Builds with tools/build_game.py into .dev/ (gitignored): the offline tools, the cached game_code
# archive (only recompiled when the recompiler, its data, runtime/ops.h or cmake/GameCode.cmake
# change) and the runtime (incremental). Game data (saves, shader/pipeline caches, settings, module
# images) uses the normal data directory, ~/.local/share/tootr, so dev runs share your saves.
#
# Environment:
#   OOT_RECOMP_NUP   the game (.nup); default: ~/.local/share/tootr/config.json's "nup",
#                    else ~/.local/share/tootr/game.nup
#   OOT_DEV_DATA     data directory override
#   JOBS             parallel compile jobs (default: picked from CPU count and memory)
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
data="${OOT_DEV_DATA:-${XDG_DATA_HOME:-$HOME/.local/share}/tootr}"
mode=run
case "${1:-}" in build|run|build-run) mode="$1"; shift ;; esac
[[ "$mode" == run && ! -x "$root/.dev/app/tootr" ]] && mode=build-run
[[ "${1:-}" == "--" ]] && shift

nup="${OOT_RECOMP_NUP:-}"
if [[ -z "$nup" && -f "$data/config.json" ]]; then
    nup="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("nup",""))' "$data/config.json")"
fi
[[ -z "$nup" && -e "$data/game.nup" ]] && nup="$(readlink -f "$data/game.nup")"

if [[ "$mode" == build || "$mode" == build-run ]]; then
    args=(--data "$data" --prefix "$root/.dev/app" --work "$root/.dev/build" --debug)
    [[ -n "$nup" ]] && args+=(--nup "$nup")
    [[ -n "${JOBS:-}" ]] && args+=(-j "$JOBS")
    python3 "$root/tools/build_game.py" "${args[@]}"
fi
if [[ "$mode" == run || "$mode" == build-run ]]; then
    [[ -n "$nup" ]] && export OOT_RECOMP_NUP="$nup"
    export OOT_RECOMP_DATA="$data"
    # (the output is also kept in .dev/logs/last-run.log, for reading crashes afterwards)
    mkdir -p "$root/.dev/logs"
    "$root/.dev/app/tootr" "$@" 2>&1 | tee "$root/.dev/logs/last-run.log"
    exit "${PIPESTATUS[0]}"
fi
