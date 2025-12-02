#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from collections import Counter
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--step-ms", type=int, default=40)
parser.add_argument("--out", required=True)
args = parser.parse_args()

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
with open(args.manifest, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line.strip()))

total_frames = 0
total_switch = 0
class_counts = Counter()
for r in rows:
    ms = r["mouth_steps"]
    total_frames += len(ms)
    for x in ms:
        class_counts[int(x)] += 1
    for a, b in zip(ms, ms[1:]):
        if a != b:
            total_switch += 1

duration_sec = total_frames * (args.step_ms / 1000)
switch_per_sec = total_switch / duration_sec if duration_sec > 0 else 0

probs = {k: v / total_frames for k, v in class_counts.items()}

summary = {
    "manifest": args.manifest,
    "total_frames": total_frames,
    "total_switch": total_switch,
    "switch_per_sec": switch_per_sec,
    "class_counts": dict(class_counts),
    "class_probs": probs,
}

with open(out_dir / "mouth_stats.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

pd.DataFrame([
    {"cls": k, "count": v, "ratio": probs[k]} for k, v in probs.items()
]).to_csv(out_dir / "mouth_stats.csv", index=False)

print(json.dumps(summary, indent=2, ensure_ascii=False))
