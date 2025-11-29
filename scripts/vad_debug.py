# scripts/vad_debug.py

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from src.m3p.vad.energy import run_energy_vad


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Energy-based VAD debug runner")
    p.add_argument("wav_path", type=str, help="入力 wav ファイルパス")
    p.add_argument(
        "--config",
        type=str,
        default="configs/vad.default.yaml",
        help="VAD 設定ファイル (YAML)",
    )
    p.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="オプション: セッションIDを上書き",
    )
    p.add_argument(
        "--utt-id",
        type=str,
        default=None,
        help="オプション: 発話IDを上書き",
    )
    p.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="出力JSONパス。未指定なら out/vad_debug/<wav名>.vad.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wav_path = Path(args.wav_path)

    # 設定ロード
    with open(args.config, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    if args.session_id is not None:
        cfg["session_id"] = args.session_id
    if args.utt_id is not None:
        cfg["utt_id"] = args.utt_id

    # VAD 実行
    result = run_energy_vad(str(wav_path), cfg)

    # 出力先決定
    if args.out_json is not None:
        out_path = Path(args.out_json)
    else:
        out_dir = Path("out") / "vad_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{wav_path.stem}.vad.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[VAD] wrote result to: {out_path}")


if __name__ == "__main__":
    main()
