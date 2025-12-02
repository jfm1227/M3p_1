import argparse, yaml, json, numpy as np
from sklearn.metrics import f1_score

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def first_onset(seq):
    # 最初に 0 (close) 以外が現れるフレーム = 発声開始
    for i, s in enumerate(seq):
        if s != 0:
            return i
    return None

def switches(seq):
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)

def evaluate_sample(y_true, y_pred, step_ms):
    # f1
    f1 = f1_score(y_true, y_pred, average="macro")

    # switch_per_sec
    sw = switches(y_pred)
    dur_sec = len(y_pred) * (step_ms / 1000)
    sw_ps = sw / dur_sec if dur_sec > 0 else 0.0

    # timing mae ms
    t0 = first_onset(y_true)
    t1 = first_onset(y_pred)
    if t0 is None or t1 is None:
        mae = 0.0
    else:
        mae = abs(t0 - t1) * step_ms

    return f1, sw_ps, mae

def main(cfg, gt_path, pred_path, report_path):
    step_ms = cfg["step_ms"]
    gts = load_jsonl(gt_path)
    preds = load_jsonl(pred_path)

    f1s, sws, maes = [], [], []

    for gt, pred in zip(gts, preds):
        y_true = gt["mouth_steps"]
        y_pred = pred["mouth_steps_pred"]
        f1, sw_ps, mae = evaluate_sample(y_true, y_pred, step_ms)
        f1s.append(f1)
        sws.append(sw_ps)
        maes.append(mae)

    summary = {
        "f1_macro": float(np.mean(f1s)),
        "switch_per_sec": float(np.mean(sws)),
        "timing_mae_ms": float(np.mean(maes))
    }

    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {report_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, "r", encoding="utf-8"))
    main(cfg, a.gt, a.pred, a.report)
