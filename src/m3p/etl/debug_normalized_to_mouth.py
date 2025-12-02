# src/m3p/etl/debug_normalized_to_mouth.py

from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


# ==============================
# かな → 母音 → mouth6 ID
# ==============================

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
    for v, chars in VOWEL_TABLE.items():
        if ch in chars:
            return v
    return None


def word_to_mouth_with_char(word: str) -> Tuple[int, str]:
    """
    normalized.json の words[*].word から
    - mouth6 ID
    - 対応に使った文字（代表文字）
    を返す。
    """
    # 語末から逆順に走査して、かな文字を拾う
    for ch in reversed(word):
        v = char_to_vowel(ch)
        if v is not None:
            mid = VOWEL_TO_MOUTH_ID.get(v, 0)
            return mid, ch
    # かなが見つからない場合は close 扱い
    return 0, ""


def load_normalized(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_intervals_with_char(
    norm: Dict[str, Any]
) -> List[Tuple[int, int, int, str, str]]:
    """
    normalized.json から
    (start_ms, end_ms, mouth_id, char_used, word_text)
    のリストを作る。
    """
    out: List[Tuple[int, int, int, str, str]] = []
    segments = norm.get("segments", [])
    for seg in segments:
        words = seg.get("words", [])
        for w in words:
            start_ms = int(w.get("start_ms", 0))
            end_ms = int(w.get("end_ms", start_ms))
            word = str(w.get("word", ""))

            mouth_id, ch = word_to_mouth_with_char(word)
            out.append((start_ms, end_ms, mouth_id, ch, word))

    # start_ms でソート
    out.sort(key=lambda x: x[0])
    return out


def build_debug_rows(
    intervals: List[Tuple[int, int, int, str, str]],
    step_ms: int = 40,
) -> List[Dict[str, str]]:
    """
    40ms グリッドで、
      t_ms, char_used, word_text, mouth_id
    をならべたデバッグ用テーブルを返す。
    """
    rows: List[Dict[str, str]] = []
    if not intervals:
        return rows

    max_end = max(end for (_, end, _, _, _) in intervals)
    n_frames = max(1, (max_end + step_ms - 1) // step_ms)

    for i in range(n_frames):
        t_ms = i * step_ms
        mouth_id = 0
        char_used = ""
        word_text = ""

        for start, end, mid, ch, w in intervals:
            if start <= t_ms < end:
                mouth_id = mid
                char_used = ch
                word_text = w
                break

        rows.append(
            {
                "t_ms": str(t_ms),
                "char": char_used,
                "word": word_text,
                "mouth_id": str(mouth_id),
            }
        )

    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Debug view: normalized.json -> mouth6 per 40ms frame"
    )
    ap.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="path to *_normalized.json (e.g. in/day7/transcript/1_0_2_normalized.json)",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="path to output TSV or CSV file (e.g. out/day7/debug/1_0_2.debug.tsv)",
    )
    ap.add_argument(
        "--step_ms",
        type=int,
        default=40,
        help="frame step in ms (default: 40 = 25fps)",
    )
    ap.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="if >0, limit number of rows in output",
    )

    args = ap.parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    norm = load_normalized(in_path)
    intervals = collect_intervals_with_char(norm)
    rows = build_debug_rows(intervals, step_ms=args.step_ms)

    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # タブ区切りにしておく（Excel / 表計算に貼りやすい）
    with out_path.open("w", encoding="utf-8") as f:
        f.write("t_ms\tchar\tword\tmouth_id\n")
        for r in rows:
            f.write(
                f"{r['t_ms']}\t{r['char']}\t{r['word']}\t{r['mouth_id']}\n"
            )

    print(
        f"[debug_normalized_to_mouth] wrote {len(rows)} rows to {out_path}"
    )


if __name__ == "__main__":
    main()
