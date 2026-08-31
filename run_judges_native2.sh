#!/bin/bash
cd "$(dirname "$0")/value-leakage"
set -a; source ../.env; set +a
for M in glm-5p2_20260815_030703 qwen3.5-122b-a10b_20260815_030702 claude-opus-4-7_20260815_042213 inkling_20260815_030703 deepseek-v4-flash-0731_20260815_030703; do
  PYTHONPATH=src ../.venv/bin/python3 -m value_leakage.judge --run_dir runs/$M --kind estimates --model claude-sonnet-5 --max_concurrent 5
done
echo NATIVE_JUDGES2_DONE
