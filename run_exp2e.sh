#!/bin/bash
cd "$(dirname "$0")"
PY=.venv/bin/python3
COMMON="--model deepseek-v4-flash-0731_20260815_030703 --provider nebius --model-id deepseek-ai/DeepSeek-V4-Flash-0731 --template deepseek --n 30 --traces 15 --conc 30"
$PY analysis/exp2_resample.py --cond below_good --arm B $COMMON
$PY analysis/exp2_resample.py --cond below_good --arm B --plain $COMMON
$PY analysis/exp2_resample.py --cond below_good --arm D $COMMON
$PY analysis/exp2_resample.py --cond below_good --arm D --plain $COMMON
echo RUNNER_E_DONE
