#!/bin/bash
cd "$(dirname "$0")/value-leakage"
set -a; source ../.env; set +a
for M in inkling-small_20260815_192811 kimi-k3_20260815_030702 minimax-m3_20260815_030703 deepseek-v4-pro-0813_20260815_030703 qwen3p8-2p4t-a95b_20260815_030703; do
  PYTHONPATH=src ../.venv/bin/python3 -m value_leakage.judge --run_dir runs/$M --kind estimates --model claude-sonnet-5 --max_concurrent 5
done
echo NATIVE_JUDGES_DONE
