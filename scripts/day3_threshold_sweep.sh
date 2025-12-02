#!/usr/bin/env bash
set -euo pipefail

PRESETS=("safe" "mid" "aggr")
CSV_LIST=("1_0_1" "1_1_1" "3_1_1")

for p in "${PRESETS[@]}"; do
  echo "===================================================="
  echo " ▶ DLC→pose 生成 (${p})"
  echo "===================================================="
  for csv in "${CSV_LIST[@]}"; do
    cfg="configs/pose_${csv}_${p}.yaml"
    echo "  - ${cfg}"
    python -m src.m3p.etl.dlc_to_pose --config "${cfg}"
  done

  echo "---- build_trainset (${p}) -------------------------"
  python -m src.m3p.etl.build_trainset \
    --config "configs/day3_${p}.yaml"

  echo "---- mouth6分析 (${p}) -----------------------------"
  python scripts/analyze_mouth_stats.py \
    --manifest "out/day3/${p}/train_samples.jsonl" \
    --step-ms 40 \
    --out "out/day3/${p}/"
done

echo "=== Day3 完了 ==="
