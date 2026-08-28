#!/usr/bin/env bash

# Shared environment for scripts/mac jobs.
MAC_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CODE_PATH="${CODE_PATH:-$(cd "$MAC_SCRIPT_DIR/../.." && pwd)}"
export PYTHON_ENV_DIR="${PYTHON_ENV_DIR:-$CODE_PATH/.venv/bin}"
export PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV_DIR/python}"
export PROJ_DIR="${PROJ_DIR:-$CODE_PATH/temp}"
