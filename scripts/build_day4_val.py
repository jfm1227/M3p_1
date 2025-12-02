# scripts/build_day4_val.py
import json, os
from collections import Counter

TRAIN_PATH = "out/day3/aggr/train_samples.jsonl"
VAL_OLD_PATH = "out/day3/aggr/val_samples.jsonl"

OUT_DIR = "out/day4"
TRAIN_NEW_PATH = os.path.join(OUT_DIR, "train_samples.v2.jsonl")
VAL_NEW_PATH = os.path.join(OUT_DIR, "val_samples.v2.jsonl")

TOP_K_VAL = 8  # 新しい val に使うサンプル数（必要に応じて変更可）

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def save_jsonl(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for js in items:
            f.write(json.dumps(js, ensure_ascii=False) + "\n")

def score_sample(js):
    steps = js["mouth_steps"]
    uniq = len(set(steps))
    switches = sum(1 for a, b in zip(steps, steps[1:]) if a != b)
    length = len(steps)
    # 「ラベルの種類数」→「スイッチ数」→「長さ」でソート
    return (uniq, switches, length)

def summarize(name, items):
    cnt = Counter()
    for js in items:
        cnt.update(js["mouth_steps"])
    total = sum(cnt.values())
    print(f"--- {name} ---")
    print(f"  samples: {len(items)}, frames: {total}")
    for k in sorted(cnt):
        print(f"    label {k}: {cnt[k]} ({cnt[k]/total:.3f})")

def main():
    train = load_jsonl(TRAIN_PATH)
    val_old = load_jsonl(VAL_OLD_PATH)  # 「第1話」1文だけのはず

    all_samples = train + val_old

    # スコア順に並べる（バリエーションが多い順）
    ranked = sorted(all_samples, key=score_sample, reverse=True)

    val_new = ranked[:TOP_K_VAL]
    train_new = ranked[TOP_K_VAL:]

    save_jsonl(TRAIN_NEW_PATH, train_new)
    save_jsonl(VAL_NEW_PATH, val_new)

    print(f"Wrote: {TRAIN_NEW_PATH}  (train_new)")
    print(f"Wrote: {VAL_NEW_PATH}   (val_new)")

    summarize("train_new", train_new)
    summarize("val_new", val_new)

if __name__ == "__main__":
    main()
