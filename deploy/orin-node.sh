#!/usr/bin/env bash
# Push this checkout to a LocalASR compute node and start it.
#
# Deliberately not a container: the node has to reach the GPU through the Tegra driver
# stack, and on JetPack that is far easier from the host than through a runtime.
#
# Nothing here needs root. `--no-dev` is what keeps PySide6, onnxruntime, soxr and
# sounddevice off the node — it has no display and no microphone, and building those
# wheels on aarch64 is minutes of CPU time spent on code that would never run.
set -euo pipefail

HOST="${1:-user@asr-node.local}"
REMOTE_DIR="${REMOTE_DIR:-~/Dev/LocalASR}"
PORT="${LOCALASR_NODE_PORT:-8090}"
NODE_PYTHON="${NODE_PYTHON:-3.12}"
# Where the node finds llama-server. On a Jetson this is a locally built CUDA binary
# rather than a downloaded release, so it is never on PATH and never vendored.
LLAMA_DIR="${LLAMA_DIR:-\$HOME/Dev/llama.cpp/build/bin}"

# Measured on this LAN: PyPI serves at ~5.8 MB/s and GitHub release assets at ~9 MB/s,
# but `releases.astral.sh` — where uv fetches its own managed Python builds — runs at
# ~125 KB/s, some 70x slower for byte-identical content. No package mirror is needed;
# only that one host is the problem. Pointing uv straight at the upstream GitHub
# releases avoids it for anyone who does want a managed interpreter. The node does not:
# it uses JetPack's own 3.12, which is why the >=3.11 floor was worth lowering.
export UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR:-https://github.com/astral-sh/python-build-standalone/releases/download}"
TOKEN="${LOCALASR_NODE_TOKEN:?set LOCALASR_NODE_TOKEN, e.g. \$(openssl rand -hex 16)}"

echo "==> syncing source to ${HOST}:${REMOTE_DIR}"
# Models are excluded on purpose: they are pinned by revision and sha256, so the node
# fetches its own rather than trusting whatever happened to be on the laptop.
rsync -az --delete \
  --exclude '.git' --exclude '.venv' --exclude 'models' --exclude '__pycache__' \
  --exclude '*.pyc' --exclude '.pytest_cache' --exclude 'samples' \
  --exclude '.python-version' \
  ./ "${HOST}:${REMOTE_DIR}/"
# `.python-version` is excluded deliberately: it pins the *developer's* interpreter
# (3.13). Shipped to the node it makes uv fetch a whole aarch64 build over a slow link
# and ignore the venv already built against JetPack's 3.12 — which is precisely the
# cost the >=3.11 floor was lowered to avoid.

echo "==> installing (node role only)"
# --no-dev has to be on *every* uv invocation, not just the sync. The dev group pulls
# localasr[capture,gui,node] so that `uv run pytest` works on a development machine, and
# a bare `uv run` on the node re-resolves with it — quietly installing PySide6,
# onnxruntime, soxr and sounddevice onto a headless Jetson and taking the venv from
# ~80 MB to 800 MB.
# --python 3.12 uses JetPack's own interpreter. Without it uv honours .python-version
# (3.13, pinned for the desktop) and downloads a whole aarch64 build over what is
# usually a slow link — which is the cost the >=3.11 floor exists to avoid.
ssh "${HOST}" "cd ${REMOTE_DIR} && PATH=\$HOME/.local/bin:\$PATH \
  UV_PYTHON_INSTALL_MIRROR='${UV_PYTHON_INSTALL_MIRROR}' \
  uv sync --no-dev --extra node --python ${NODE_PYTHON}"

echo "==> stopping any previous node"
# `[l]ocalasr-node` rather than `localasr-node`: this very ssh command line contains the
# name, so the plain pattern matches the shell that is about to do the starting and
# kills it before it gets there. The bracket matches the running node but not the
# literal text in our own argv.
ssh "${HOST}" "pkill -f '[l]ocalasr-node' 2>/dev/null; exit 0"

echo "==> starting node on port ${PORT}"
ssh "${HOST}" "cd ${REMOTE_DIR} && \
  PATH=\$HOME/.local/bin:\$PATH \
  LOCALASR_NODE_BIND=0.0.0.0 \
  LOCALASR_NODE_PORT=${PORT} \
  LOCALASR_NODE_TOKEN=${TOKEN} \
  LOCALASR_DATA_DIR=${REMOTE_DIR} \
  LOCALASR_LLAMA_DIR=${LLAMA_DIR} \
  setsid uv run --no-dev --python ${NODE_PYTHON} localasr-node \
    < /dev/null > ${REMOTE_DIR}/node.log 2>&1 &" || true

echo "==> waiting for /healthz"
NODE_HOST="${HOST#*@}"
for _ in $(seq 1 30); do
  if curl -sf -m 3 "http://${NODE_HOST}:${PORT}/healthz" >/dev/null; then
    echo "    up: http://${NODE_HOST}:${PORT}"
    curl -s -H "Authorization: Bearer ${TOKEN}" "http://${NODE_HOST}:${PORT}/api/v1/models"
    echo
    exit 0
  fi
  sleep 2
done

echo "    node did not answer; last log lines:" >&2
ssh "${HOST}" "tail -20 ${REMOTE_DIR}/node.log" >&2
exit 1
