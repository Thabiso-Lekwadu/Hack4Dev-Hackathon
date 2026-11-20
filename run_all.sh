#!/usr/bin/env bash
# Executes the three solution notebooks in order (set H4D_QUICK=1 for a short smoke run).
set -e
cd "$(dirname "$0")"
for nb in 5-Improved_Feature_Engineering 6-Improved_Modelling_and_Transfer 7-Decision_Evidence_and_Leakage_Audit; do
  echo "START $nb $(date +%T)"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$nb.ipynb"
  echo "OK $nb $(date +%T)"
done
