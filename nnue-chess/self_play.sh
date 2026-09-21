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
SEARCH_DEPTH=2
MAX_PLIES=150
NAME=selfplay
ARCH=""
BUILD_DIR=build

while [[ $# -gt 0 ]]; do
  case "$1" in
    --games) GAMES="$2"; shift 2 ;;
    --search-depth) SEARCH_DEPTH="$2"; shift 2 ;;
    --max-plies) MAX_PLIES="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --report)
      cmake --build "$BUILD_DIR" -j --target self_play >/dev/null
      exec "$BUILD_DIR/self_play" --report ;;
    -h|--help)
      echo "Usage: $0 --games N --search-depth N --max-plies N --name NAME [--arch ACC,H1,H2,H3] [--build-dir DIR]"
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
RUN_ARGS=(--games "$GAMES" --search-depth "$SEARCH_DEPTH" --max-plies "$MAX_PLIES" --name "$NAME")
[[ -n "$ARCH" ]] && RUN_ARGS+=(--arch "$ARCH")
exec "$BUILD_DIR/self_play" "${RUN_ARGS[@]}"
