# src/m3p/etl/dlc_to_pose.py

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MOUTHS = ["close", "a", "i", "u", "e", "o"]
M2ID = {m: i for i, m in enumerate(MOUTHS)}


def dist(x1, y1, x2, y2):
    return float(math.hypot(x1 - x2, y1 - y2))


def compute_mouth_features(row, COL):
    """
    1フレーム分から正規化済み open_h, mouth_w, 顔幅 fw を計算する。
    """
    fw = max(
        1e-6,
        dist(row[COL["flx"]], row[COL["fly"]], row[COL["frx"]], row[COL["fry"]]),
    )
    open_h = dist(row[COL["ulx"]], row[COL["uly"]], row[COL["llx"]], row[COL["lly"]]) / fw
    mouth_w = dist(row[COL["mlx"]], row[COL["mly"]], row[COL["mrx"]], row[COL["mry"]]) / fw
    return open_h, mouth_w, fw


def estimate_mouth6(row, COL, tau, rho):
    """
    row: DataFrame の1行（Series）
    COL: dlc_columns (dict)
    tau: (τ1, τ2, τ3)
    rho: (ρ1, ρ2, ρ3)
    """
    open_h, mouth_w, fw = compute_mouth_features(row, COL)

    τ1, τ2, τ3 = tau
    ρ1, ρ2, ρ3 = rho

    # ルールベース口形分類（M3prime版と同じロジック）
    if open_h < τ1:
        return "close"
    if open_h > τ3 and mouth_w < ρ2:
        return "o"
    if open_h > τ3:
        return "a"
    if open_h < τ2 and mouth_w > ρ3:
        return "i"
    if mouth_w < ρ1:
        return "u"
    return "e"


def median_filter(ids, k):
    """簡易メディアンフィルタ"""
    if k <= 1:
        return ids
    out = []
    buf = []
    for x in ids:
        buf.append(x)
        if len(buf) > k:
            buf.pop(0)
        out.append(int(np.median(buf)))
    return out


def hysteresis_filter(ids, low_keep=1, high_keep=1):
    """
    ヒステリシスフィルタ：
    - run<=low_keep の短いノイズは前のIDに吸収
    """
    if not ids:
        return ids
    out = [ids[0]]
    run = 1
    for i in range(1, len(ids)):
        if ids[i] == out[-1]:
            run += 1
            out.append(ids[i])
            continue
        if run <= low_keep and i + 1 < len(ids) and ids[i + 1] == out[-1]:
            out.append(out[-1])
            run += 1
        else:
            out.append(ids[i])
            run = 1
    return out


def compute_t_ms(df: pd.DataFrame, COL: dict) -> pd.Series:
    """
    時刻列 t_ms を生成する。
    優先順位:
      1) COL["t_ms"] が df にあればそれを使う
      2) COL["timestamp"] があれば「秒」とみなして ms に変換
      3) どちらもなければ frame + fps から算出
    """
    # 1) 既存 t_ms 列
    if "t_ms" in COL and COL["t_ms"] in df.columns:
        return df[COL["t_ms"]].astype(float).round().astype(int)

    # 2) timestamp[秒]
    if "timestamp" in COL and COL["timestamp"] in df.columns:
        return (df[COL["timestamp"]].astype(float) * 1000.0).round().astype(int)

    # 3) frame + fps
    fps = COL.get("fps")
    frame_col = COL.get("frame", "frame")
    assert fps, "dlc_columns.fps が設定されていません（t_ms/timestamp が無い場合に必要）"
    assert frame_col in df.columns, f"frame 列 '{frame_col}' が見つかりません"

    return (df[frame_col].astype(float) * 1000.0 / float(fps)).round().astype(int)


def debug_stats_open_width(open_h_list, mouth_w_list, tau, rho, prefix="[debug]"):
    """
    open_h / mouth_w の分布と τ/ρ に対する位置関係をざっくり出力。
    """
    oh = np.array(open_h_list, dtype=float)
    mw = np.array(mouth_w_list, dtype=float)

    if oh.size == 0:
        print(f"{prefix} no frames after filtering")
        return

    def q(x, qs):
        return {str(q): float(np.quantile(x, q)) for q in qs}

    print(f"{prefix} open_h stats: min={oh.min():.4f}, max={oh.max():.4f}, mean={oh.mean():.4f}")
    print(f"{prefix} open_h quantiles:", q(oh, [0.1, 0.25, 0.5, 0.75, 0.9]))
    print(f"{prefix} mouth_w stats: min={mw.min():.4f}, max={mw.max():.4f}, mean={mw.mean():.4f}")
    print(f"{prefix} mouth_w quantiles:", q(mw, [0.1, 0.25, 0.5, 0.75, 0.9]))

    τ1, τ2, τ3 = tau
    ρ1, ρ2, ρ3 = rho

    # τ によるゾーン比率
    zones_open = {
        f"< τ1 ({τ1:.3f})": float((oh < τ1).mean()),
        f"[τ1, τ2) ({τ1:.3f}-{τ2:.3f})": float(((oh >= τ1) & (oh < τ2)).mean()),
        f"[τ2, τ3) ({τ2:.3f}-{τ3:.3f})": float(((oh >= τ2) & (oh < τ3)).mean()),
        f">= τ3 ({τ3:.3f})": float((oh >= τ3).mean()),
    }
    zones_width = {
        f"< ρ1 ({ρ1:.3f})": float((mw < ρ1).mean()),
        f"[ρ1, ρ2) ({ρ1:.3f}-{ρ2:.3f})": float(((mw >= ρ1) & (mw < ρ2)).mean()),
        f"[ρ2, ρ3) ({ρ2:.3f}-{ρ3:.3f})": float(((mw >= ρ2) & (mw < ρ3)).mean()),
        f">= ρ3 ({ρ3:.3f})": float((mw >= ρ3).mean()),
    }

    print(f"{prefix} open_h zone ratios:", zones_open)
    print(f"{prefix} mouth_w zone ratios:", zones_width)


def debug_stats_mouth_ids(raw_ids, smoothed_ids, prefix="[debug]"):
    """
    口型IDのクラス分布を出力（フィルタ前後）。
    """

    def hist(ids):
        if not ids:
            return {}
        arr = np.array(ids, dtype=int)
        counts = dict(zip(*np.unique(arr, return_counts=True)))
        total = float(arr.size)
        return {
            MOUTHS[i]: {"count": int(c), "ratio": float(c / total)}
            for i, c in counts.items()
        }

    print(f"{prefix} mouth6 histogram (raw):", hist(raw_ids))
    print(f"{prefix} mouth6 histogram (smoothed):", hist(smoothed_ids))


def main(cfg):
    paths = cfg["paths"]
    COL = cfg["dlc_columns"]  # default.yaml で DLC 列名を指定
    step_ms = cfg["step_ms"]
    like_thr = cfg.get("likelihood_threshold", 0.0)
    tau = cfg["tau"]
    rho = cfg["rho"]

    median_k = cfg.get("median_k", 5)
    hys_low_keep = cfg.get("hys_low_keep", 1)
    hys_high_keep = cfg.get("hys_high_keep", 1)

    debug = cfg.get("debug", False)

    os.makedirs(Path(paths["pose_timeline"]).parent, exist_ok=True)

    df = pd.read_csv(paths["dlc_csv"])
    print(f"[dlc_to_pose] loaded CSV: {paths['dlc_csv']}  (rows={len(df)})")

    # --- t_ms 列の生成 ---
    df["t_ms"] = compute_t_ms(df, COL)
    df = df.sort_values("t_ms").copy()

    # --- likelihood フィルタ（任意）---
    lk_cols = [
        COL.get("lk_ul"),
        COL.get("lk_ll"),
        COL.get("lk_ml"),
        COL.get("lk_mr"),
    ]
    lk_cols = [c for c in lk_cols if c and c in df.columns]
    if lk_cols and like_thr > 0:
        lk = df[lk_cols].min(axis=1)
        before = len(df)
        df = df.loc[(lk >= like_thr) | (lk.isna())].copy()
        after = len(df)
        print(f"[dlc_to_pose] likelihood filter: {before} -> {after} rows (thr={like_thr})")
    else:
        print(f"[dlc_to_pose] likelihood filter: skipped (lk_cols={lk_cols}, thr={like_thr})")

    if len(df) == 0:
        print("[dlc_to_pose] no frames left after filtering, writing empty timeline.")
        obj = {"meta": {"step_ms": step_ms}, "timeline": []}
        with open(paths["pose_timeline"], "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return

    # --- 口形推定の前に open_h / mouth_w を計算（デバッグ用） ---
    open_h_list = []
    mouth_w_list = []
    for _, row in df.iterrows():
        oh, mw, _ = compute_mouth_features(row, COL)
        open_h_list.append(oh)
        mouth_w_list.append(mw)

    if debug:
        debug_stats_open_width(open_h_list, mouth_w_list, tau, rho)

    # --- 口形推定（raw ids）---
    mouths = [estimate_mouth6(row, COL, tau, rho) for _, row in df.iterrows()]
    raw_ids = [M2ID[m] for m in mouths]

    # スムージング
    ids = median_filter(raw_ids, median_k)
    ids = hysteresis_filter(ids, low_keep=hys_low_keep, high_keep=hys_high_keep)

    # デバッグ用 mouth6 ヒストグラム
    debug_stats_mouth_ids(raw_ids, ids)

    # --- タイムライン圧縮 ---
    timeline = []
    prev = None
    for t, i in zip(df["t_ms"].tolist(), ids):
        if prev is None or i != prev:
            timeline.append({"t_ms": int(t), "mouth6": MOUTHS[i]})
            prev = i

    obj = {"meta": {"step_ms": step_ms}, "timeline": timeline}
    with open(paths["pose_timeline"], "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    print(f"[dlc_to_pose] Wrote: {paths['pose_timeline']} (events={len(timeline)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
