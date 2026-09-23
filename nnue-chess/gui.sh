#!/usr/bin/env bash
# Starts the training GUI (see gui/server.py) - stdlib-only Python, no pip
# install needed. Open the printed URL in a browser (WSL2 forwards
# localhost to Windows automatically).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec python3 gui/server.py "$@"
