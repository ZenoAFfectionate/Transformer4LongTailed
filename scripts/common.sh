#!/usr/bin/env bash
# Shared environment variables for all MoE4ViT train scripts.
# Source this file from each wrapper; override via environment before calling.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU="${GPU:-0}"
DATA_ROOT="${DATA_ROOT:-/home/kemove/data}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results}"
WORKERS="${WORKERS:-8}"

export MOE4VIT_RESULT_DIR="${RESULT_ROOT}"

# Re-export for downstream processes.
export REPO_ROOT SCRIPT_DIR GPU DATA_ROOT RESULT_ROOT WORKERS
