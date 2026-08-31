#!/bin/bash
cd "$(dirname "$0")"
PY=.venv/bin/python3
$PY analysis/exp3_confession.py --model glm --cond below_good --n 40 --conc 20
$PY analysis/exp3_confession.py --model glm --cond above_good --n 40 --conc 20
until grep -q RUNNER_E_DONE exp2_out/runner_e.log 2>/dev/null; do sleep 120; done
$PY analysis/exp3_confession.py --model deepseek --cond below_good --n 40 --conc 20
$PY analysis/exp3_confession.py --model deepseek --cond above_good --n 40 --conc 20
echo EXP3_DONE
