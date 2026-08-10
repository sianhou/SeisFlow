#!/usr/bin/env bash

# Shared environment for scripts/hwgpu training and reconstruction jobs.
export CODE_PATH="${CODE_PATH:-/hwdata/24ydz3d/deeplearning/SeisFlow}"
export PYTHON_ENV_DIR="${PYTHON_ENV_DIR:-/hwdata/24ydz3d/deeplearning/environments/py313/bin}"
export PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV_DIR/python3.13}"
export TORCHRUN_BIN="${TORCHRUN_BIN:-$PYTHON_ENV_DIR/torchrun}"
export PROJ_DIR="${PROJ_DIR:-/hwdata/24ydz3d/deeplearning/PROJ}"
export MASTER="${MASTER:-$(hostname)}"
if [[ -z "${MASTER_ADDR:-}" ]]; then
    MASTER_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
    MASTER_ADDR="${MASTER_ADDR:-$MASTER}"
fi
export MASTER_ADDR
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MASTER_PORT="${MASTER_PORT:-29500}"
