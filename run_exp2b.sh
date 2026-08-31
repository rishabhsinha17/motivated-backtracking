#!/bin/bash
cd "$(dirname "$0")"
PY=.venv/bin/python3
COMMON="--model glm-5p2_20260815_030703 --provider nebius --model-id zai-org/GLM-5.2 --template glm --n 30 --traces 15 --conc 30"
$PY analysis/exp2_resample.py --cond below_good --arm B $COMMON
$PY analysis/exp2_resample.py --cond below_good --arm B --plain $COMMON
$PY analysis/exp2_resample.py --cond below_good --arm D $COMMON
$PY analysis/exp2_resample.py --cond below_good --arm D --plain $COMMON
echo RUNNER_B_DONE
