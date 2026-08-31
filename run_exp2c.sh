#!/bin/bash
cd "$(dirname "$0")"
PY=.venv/bin/python3
COMMON="--model deepseek-v4-flash-0731_20260815_030703 --provider nebius --model-id deepseek-ai/DeepSeek-V4-Flash-0731 --template deepseek --n 30 --traces 15 --conc 20"
$PY analysis/exp2_resample.py --cond below_good --arm A $COMMON
$PY analysis/exp2_resample.py --cond below_good --arm A --plain $COMMON
echo RUNNER_C_DONE
