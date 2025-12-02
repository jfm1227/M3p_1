# src/m3p/etl/normalized_to_mouth_timeline.py

from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


# ==============================
# かな → 母音 → mouth6 ID
# ==============================

# ひらがな・カタカナをざっくり 5 母音にマッピング
VOWEL_TABLE = {
    "a": set(
        "あかがさざただなはばぱまやらわぁゃゎ"
        "アカガサザタダナハバパマヤラワァャヮ"
    ),
    "i": set(
        "いきぎしじちぢにひびぴみりぃ"
        "イキギシジチヂニヒビピミリィ"
    ),
    "u": set(
        "うくぐすずつづぬふぶぷむゆるぅゅ"
        "ウクグスズツヅヌフブプムユルゥュ"
    ),
    "e": set(
        "えけげせぜてでねへべぺめれぇ"
        "エケゲセゼテデネヘベペメレェ"
    ),
    "o": set(
        "おこごそぞとどのほぼぽもよろをぉょ"
        "オコゴソゾトドノホボポモヨロヲォョ"
    ),
}

VOWEL_TO_MOUTH_ID = {
    "a": 1,
    "i": 2,
    "u": 3,
    "e": 4,
    "o": 5,
}


def char_to_vowel(ch: str) -> Optional[str]:
    """1文字から母音 ('a','i','u','e','o') を推定"""
    for v, chars in VOWEL_TABLE.items():
        if ch in chars:
            return v
    return None


def word_to_mouth_id(word: str) -> int:
    """
    normalized.json の words[*].word から mouth6 ID を決める。
    - 語末から逆順に走査して、かな文字を見つけたらその母音で決定。
    - 見つからなければ close(0) 扱い。
    """
    # 語末から走査（句読点などをスキップしたい）
    for ch in reversed(word):
        v = char_to_vowel(ch)
        if v is not None:
            return VOWEL_TO_MOUTH_ID.get(v, 0)
    # 母音が見つからない場合は「閉口」とみなす
    return 0


# ==============================
# normalized.json → mouth_timeline
# ==============================

def load_normalized(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_word_intervals(norm: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    """
    normalized.json から (start_ms, end_ms, mouth_id) のリストを作る。
    """
    intervals: List[Tuple[int, int, int]] = []

    segments = norm.get("segments", [])
    for seg in segments:
        words = seg.get("words", [])
        for w in words:
            start_ms = int(w.get("start_ms", 0))
            end_ms = int(w.get("end_ms", start_ms))
            word = str(w.get("word", ""))

            mouth_id = word_to_mouth_id(word)
            intervals.append((start_ms, end_ms, mouth_id))

    # 念のため start_ms でソート
    intervals.sort(key=lambda x: x[0])
    return intervals


def build_mouth_timeline(
    intervals: List[Tuple[int, int, int]],
    step_ms: int = 40,
) -> List[Dict[str, int]]:
    """
    (start_ms, end_ms, mouth_id) のリストから
    40ms グリッドの mouth_timeline を生成する。
    """
    if not intervals:
        return []

    # タイムラインの最大 ms を求める
    max_end = max(end for (_, end, _) in intervals)
    # 40ms グリッドのフレーム数
    n_frames = max(1, (max_end + step_ms - 1) // step_ms)

    frames: List[Dict[str, int]] = []

    # 単純に O(T * N) で間に合う想定
    for i in range(n_frames):
        t_ms = i * step_ms
        mouth_id = 0  # デフォルト close

        for start, end, mid in intervals:
            if start <= t_ms < end:
                mouth_id = mid
                break

        frames.append({"t_ms": int(t_ms), "mouth_id": int(mouth_id)})

    return frames


def normalized_to_mouth_timeline(
    in_path: Path,
    out_path: Path,
    step_ms: int = 40,
) -> None:
    norm = load_normalized(in_path)
    intervals = collect_word_intervals(norm)
    frames = build_mouth_timeline(intervals, step_ms=step_ms)

    meta = norm.get("meta", {})
    audio = meta.get("source_audio", "")

    out_js = {
        "audio": audio,
        "step_ms": step_ms,
        "frames": frames,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_js, f, ensure_ascii=False, indent=2)

    print(f"[normalized_to_mouth_timeline] wrote {out_path} frames={len(frames)}")


def main():
    ap = argparse.ArgumentParser(
        description="Convert normalized.json to mouth_timeline.json (mouth6, step_ms=40)"
    )
    ap.add_argument("--in", dest="in_path", required=True,
                    help="path to normalized.json")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="path to output mouth_timeline.json")
    ap.add_argument("--step_ms", type=int, default=40,
                    help="frame step in milliseconds (default: 40 = 25fps)")

    args = ap.parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    normalized_to_mouth_timeline(in_path, out_path, step_ms=args.step_ms)


if __name__ == "__main__":
    main()
