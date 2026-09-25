#!/bin/bash
# Distributed launcher for supervised InfoNCE, RankNet, or LambdaLoss training.

set -euo pipefail

export FORCE_TORCHRUN=1
export NNODES=${NNODES:-1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export WANDB_PROJECT=${WANDB_PROJECT:-RELER}

if [ $# -gt 0 ]; then
  first_arg=$1
  if [[ "$first_arg" == *.yaml || "$first_arg" == *.yml || "$first_arg" == *.json ]]; then
    if [ ! -f "$first_arg" ]; then
      echo "Config file not found: $first_arg" >&2
      exit 1
    fi
  fi
fi

torchrun \
  --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" \
  --module reler.cli supervised \
  "$@"
