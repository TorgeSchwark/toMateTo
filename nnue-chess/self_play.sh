#!/usr/bin/env bash
# Convenience wrapper around the self-play trainer (see
# training/self_play.cpp): makes the NNUE architecture *feel* like an
# ordinary runtime flag even though it is actually a compile-time template
# parameter, by reconfiguring + rebuilding CMake whenever --arch changes.
# Mirrors train.sh (the Stockfish-supervised trainer) - see that file too.
#
# Usage:
#   ./self_play.sh --games 500 --search-depth 2 --max-plies 150 --name sp_net [--arch 256,128,32,32] [--build-dir build]
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

GAMES=200
TD_MODE=leaf
SEARCH_DEPTH=2
ROLLOUT_DEPTH=7
TD_LAMBDA=0.7
MAX_PLIES=150
SYNC_EVERY=10
INIT_FROM=""
LR=""
LR_HALF_LIFE=""
NAME=selfplay
ARCH=""
BUILD_DIR=build

while [[ $# -gt 0 ]]; do
  case "$1" in
    --games) GAMES="$2"; shift 2 ;;
    --td-mode) TD_MODE="$2"; shift 2 ;;
    --search-depth) SEARCH_DEPTH="$2"; shift 2 ;;
    --rollout-depth) ROLLOUT_DEPTH="$2"; shift 2 ;;
    --td-lambda) TD_LAMBDA="$2"; shift 2 ;;
    --max-plies) MAX_PLIES="$2"; shift 2 ;;
    --sync-every) SYNC_EVERY="$2"; shift 2 ;;
    --init-from) INIT_FROM="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --lr-half-life) LR_HALF_LIFE="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --report)
      cmake --build "$BUILD_DIR" -j --target self_play >/dev/null
      exec "$BUILD_DIR/self_play" --report ;;
    -h|--help)
      echo "Usage: $0 --games N [--td-mode leaf|lambda] [--search-depth N | --rollout-depth N --td-lambda F]"
      echo "          [--max-plies N] [--sync-every N] [--init-from path.nnue] --name NAME"
      echo "          [--arch ACC,H1,H2,H3] [--build-dir DIR]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

CMAKE_ARGS=(-S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release)

if [[ -n "$ARCH" ]]; then
  IFS=',' read -r ACC H1 H2 H3 <<< "$ARCH"
  if [[ -z "${H3:-}" ]]; then
    echo "Error: --arch must be ACC,H1,H2,H3 (e.g. 256,128,32,32), got: $ARCH" >&2
    exit 1
  fi
  CMAKE_ARGS+=(-DNNUE_ACC_SIZE="$ACC" -DNNUE_H1="$H1" -DNNUE_H2="$H2" -DNNUE_H3="$H3")
fi

echo "== configuring (architecture change triggers a rebuild automatically) =="
cmake "${CMAKE_ARGS[@]}"

echo "== building self_play =="
cmake --build "$BUILD_DIR" -j --target self_play

echo "== running =="
RUN_ARGS=(--games "$GAMES" --td-mode "$TD_MODE" --max-plies "$MAX_PLIES" --sync-every "$SYNC_EVERY" --name "$NAME")
if [[ "$TD_MODE" == "lambda" ]]; then
  RUN_ARGS+=(--rollout-depth "$ROLLOUT_DEPTH" --td-lambda "$TD_LAMBDA")
else
  RUN_ARGS+=(--search-depth "$SEARCH_DEPTH")
fi
[[ -n "$INIT_FROM" ]] && RUN_ARGS+=(--init-from "$INIT_FROM")
[[ -n "$LR" ]] && RUN_ARGS+=(--lr "$LR")
[[ -n "$LR_HALF_LIFE" ]] && RUN_ARGS+=(--lr-half-life "$LR_HALF_LIFE")
[[ -n "$ARCH" ]] && RUN_ARGS+=(--arch "$ARCH")
exec "$BUILD_DIR/self_play" "${RUN_ARGS[@]}"
