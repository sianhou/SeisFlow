#!/usr/bin/env bash

# Shared environment for scripts/dreamcloud jobs.
export CODE_PATH="${CODE_PATH:-/dreamdata/24yds3d/deeplearning/SeisFlow}"
export PYTHON_ENV_DIR="${PYTHON_ENV_DIR:-/dreamdata/24yds3d/deeplearning/python313/bin}"
export PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV_DIR/python3.13}"
export TORCHRUN_BIN="${TORCHRUN_BIN:-$PYTHON_ENV_DIR/torchrun}"
export PROJ_DIR="${PROJ_DIR:-/dreamdata/24yds3d/deeplearning/Temp}"
export MASTER="${MASTER:-$(hostname)}"
if [[ -z "${MASTER_ADDR:-}" ]]; then
    MASTER_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
    MASTER_ADDR="${MASTER_ADDR:-$MASTER}"
fi
export MASTER_ADDR
export NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export MASTER_PORT="${MASTER_PORT:-29500}"
