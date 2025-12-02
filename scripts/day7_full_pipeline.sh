# scripts/day7_full_pipeline.sh
#!/usr/bin/env bash
set -euo pipefail

# 1) ルートと PYTHONPATH 設定
cd /workspaces/M3p_1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "[day7] START full pipeline (build_trainset -> train -> infer -> eval)"

# 2) ETL: manifest (27本) -> train/val JSONL (v3.full)
echo "[day7] build_trainset (from configs/day7_build_trainset_full.yaml)"
python -m m3p.etl.build_trainset \
  --config configs/day7_build_trainset_full.yaml

# 3) 学習: マルチタスク版 (CSV27本ベースの中サイズ)
echo "[day7] train (configs/day7_multitask_full.yaml)"
python -m m3p.train \
  --config configs/day7_multitask_full.yaml

# 4) 推論: val セットに対して推論
VAL_JSONL="out/day7/val_samples.v3.full.jsonl"
PRED_JSONL="out/exp_day7_multitask_full/pred.val.day7.full.jsonl"
REPORT_JSON="out/exp_day7_multitask_full/report.val.day7.full.json"

echo "[day7] infer on val (${VAL_JSONL})"
python -m m3p.infer.jsonl_infer \
  --config configs/day7_multitask_full.yaml \
  --in  "${VAL_JSONL}" \
  --out "${PRED_JSONL}"

# 5) 評価: f1_macro / switch_per_sec / timing_mae_ms
echo "[day7] eval on val (gt=${VAL_JSONL}, pred=${PRED_JSONL})"
python -m m3p.eval \
  --config configs/day7_multitask_full.yaml \
  --gt   "${VAL_JSONL}" \
  --pred "${PRED_JSONL}" \
  --report "${REPORT_JSON}"

echo "[day7] DONE. Report: ${REPORT_JSON}"
