# scripts/day7_dlc_to_pose_batch.py
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from m3p.etl import dlc_to_pose


# Day7 で使う 27 本分の ID リスト
DAY7_IDS = [
    "1_0_2",
    "1_0_3",
    "1_0_4",
    "1_0_5",
    "1_0_71",
    "1_0_72",
    "1_0_73",
    "1_0_74",
    "1_0_75",
    "1_1_2",
    "1_1_3",
    "1_1_4",
    "1_1_5",
    "1_2_1",
    "1_2_2",
    "1_2_3",
    "1_2_4",
    "1_2_5",
    "2_1_1",
    "2_1_2",
    "2_1_3",
    "2_1_4",
    "2_1_5",
    "3_1_2",
    "3_1_3",
    "3_1_4",
    "3_1_5",
]


def main() -> None:
    # M3p_1 ルートを基準にする想定
    root = Path(__file__).resolve().parents[1]

    base_cfg_path = root / "configs" / "pose_day7_base.yaml"
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    for sid in DAY7_IDS:
        cfg = copy.deepcopy(base_cfg)

        # ★ CSV は in/ 直下にある前提
        dlc_csv = root / "in" / f"{sid}.csv"

        # ★ 出力は out/day7/pose/ID.pose_timeline.json
        pose_out = root / "out" / "day7" / "pose" / f"{sid}.pose_timeline.json"

        cfg["paths"]["dlc_csv"] = str(dlc_csv)
        cfg["paths"]["pose_timeline"] = str(pose_out)

        pose_out.parent.mkdir(parents=True, exist_ok=True)

        print(f"[day7] dlc_to_pose for {sid}")
        print(f"  dlc_csv      = {cfg['paths']['dlc_csv']}")
        print(f"  poseTimeline = {cfg['paths']['pose_timeline']}")

        # dlc_to_pose.py 側の main(cfg) を直接呼び出す
        dlc_to_pose.main(cfg)

    print("[day7] all done.")


if __name__ == "__main__":
    main()
