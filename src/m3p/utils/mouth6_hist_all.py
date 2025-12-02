#!/usr/bin/env python3
import os
import json
import argparse
from collections import Counter, defaultdict
from glob import glob

def to_mouth6(char: str) -> int:
    # 日本語ひらがな/カタカナに対する mouth6 マッピング（簡易）
    a_set = {"あ","か","さ","た","な","は","ま","や","ら","わ",
             "が","ざ","だ","ば","ぱ"}
    i_set = {"い","き","し","ち","に","ひ","み","り",
             "ぎ","じ","ぢ","び","ぴ"}
    u_set = {"う","く","す","つ","ぬ","ふ","む","ゆ","る",
             "ぐ","ず","づ","ぶ","ぷ"}
    e_set = {"え","け","せ","て","ね","へ","め","れ",
             "げ","ぜ","で","べ","ぺ"}
    o_set = {"お","こ","そ","と","の","ほ","も","よ","ろ",
             "ご","ぞ","ど","ぼ","ぽ"}
    
    if char in a_set: return 1
    if char in i_set: return 2
    if char in u_set: return 3
    if char in e_set: return 4
    if char in o_set: return 5
    return 0  # その他 close

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir",
                        default="in/day7/transcript",
                        help="dir with normalized.json")
    parser.add_argument("--out", default="out/mouth6_hist_all",
                        help="output dir")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = glob(os.path.join(args.in_dir, "*_normalized.json"))
    if len(files) == 0:
        print("No *_normalized.json found!")
        return

    print(f"Found {len(files)} normalized.json files")

    total_counter = Counter()
    per_file_results = {}

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cnt = Counter()
        segs = data.get("segments", [])
        for seg in segs:
            for w in seg.get("words", []):
                c = w.get("word", "")
                if not c:
                    continue
                # 1文字ずつ処理
                for ch in c:
                    m = to_mouth6(ch)
                    cnt[m] += 1
                    total_counter[m] += 1

        fname = os.path.basename(path)
        per_file_results[fname] = dict(cnt)
        print(f"Processed {fname}: {dict(cnt)}")

    # Save per-file JSON
    with open(os.path.join(args.out, "per_file_hist.json"), "w", encoding="utf-8") as f:
        json.dump(per_file_results, f, indent=2, ensure_ascii=False)

    # Save total JSON
    with open(os.path.join(args.out, "total_hist.json"), "w", encoding="utf-8") as f:
        json.dump(dict(total_counter), f, indent=2, ensure_ascii=False)

    print("\n=== TOTAL MOUTH6 HIST ===")
    print(dict(total_counter))
    print(f"\nSaved to: {args.out}/")

if __name__ == "__main__":
    main()
