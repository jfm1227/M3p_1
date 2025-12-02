# src/m3p/train.py

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from m3p.data.dataset import MouthDataset, collate, VOCAB
from m3p.model.t2mouth import Text2Mouth


def main(cfg):
    # ---- 設定の取り出し ----
    train_cfg = cfg.get("train", {})
    paths = cfg["paths"]

    # seed は cfg.seed があれば優先、なければ train.seed → それもなければ 42
    seed = cfg.get("seed", train_cfg.get("seed", 42))
    torch.manual_seed(seed)

    # device 判定
    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[train] device = {device}, seed = {seed}")

    # 出力ディレクトリ準備
    out_dir = paths["out_dir"]
    ckpt_dir = os.path.join(out_dir, "ckpt")
    metrics_dir = os.path.join(out_dir, "metrics")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # ---- データセット & DataLoader ----
    train_path = paths["train_samples"]
    ds = MouthDataset(train_path)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=train_cfg.get("num_workers", 0),
    )
    print(f"[train] train_samples = {train_path}, size = {len(ds)}")

    # ---- モデル & オプティマイザ ----
    model_cfg = cfg["model"]
    model = Text2Mouth(VOCAB, **model_cfg).to(device)

    opt = optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # fine 用 CE
    ce_fine = nn.CrossEntropyLoss(
        ignore_index=-100,
        label_smoothing=train_cfg.get("label_smoothing", 0.0),
    )

    # multitask（coarse 3クラス）を使うか？
    use_multitask = bool(model_cfg.get("multitask", False))
    if use_multitask:
        # 0:close, 1/3/5:縦系(a/u/o)→2, 2/4:横系(i/e)→1
        coarse_map = torch.tensor([0, 2, 1, 2, 1, 2], device=device)
        ce_coarse = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=train_cfg.get("coarse_label_smoothing", 0.0),
        )
        coarse_loss_weight = float(train_cfg.get("coarse_loss_weight", 0.3))
        print(
            f"[train] multitask enabled: coarse_loss_weight={coarse_loss_weight}"
        )
    else:
        coarse_map = None
        ce_coarse = None
        coarse_loss_weight = 0.0
        print("[train] multitask disabled (fine 6-class only)")

    grad_clip = train_cfg.get("grad_clip", 1.0)
    patience = train_cfg.get("early_stop_patience", 5)

    # ---- 学習ループ ----
    best = float("inf")
    no_improve = 0
    num_epochs = train_cfg["num_epochs"]

    for ep in range(num_epochs):
        model.train()
        tot_loss = 0.0
        steps = 0

        for batch in dl:
            x = batch["x"].to(device)       # [B, L]
            mask = batch["mask"].to(device) # [B, L]
            y = batch["y"].to(device)       # [B, maxT]
            T_list = batch["T_list"]        # list[int], 長さ B

            # 前向き計算
            if use_multitask:
                logits_fine, logits_coarse = model(
                    x, mask, T_list, return_coarse=True
                )  # [B, maxT, 6], [B, maxT, 3]
            else:
                logits_fine = model(x, mask, T_list)  # [B, maxT, 6]
                logits_coarse = None

            # coarse ラベルを on-the-fly で生成
            if use_multitask:
                # y: [B, maxT], 値は {0..5} or -100
                y_tmp = y.clone()
                valid = y_tmp != -100
                if valid.any():
                    # -100 を 0 に一時的に潰してから coarse_map に通す
                    y_tmp[~valid] = 0
                    y_coarse = coarse_map[y_tmp]      # [B, maxT]
                    y_coarse[~valid] = -100
                else:
                    y_coarse = y_tmp  # 全部 -100
            else:
                y_coarse = None

            # 1サンプルごとに有効長 T を切り出して CE 計算
            loss_fine = 0.0
            loss_coarse = 0.0
            n_tok = 0

            for i, T in enumerate(T_list):
                t = int(T)
                if t <= 0:
                    continue

                # fine
                loss_fine = loss_fine + ce_fine(
                    logits_fine[i, :t], y[i, :t]
                )

                # coarse
                if use_multitask:
                    loss_coarse = loss_coarse + ce_coarse(
                        logits_coarse[i, :t], y_coarse[i, :t]
                    )

                n_tok += t

            if n_tok == 0:
                continue

            loss_fine = loss_fine / n_tok
            if use_multitask:
                loss_coarse = loss_coarse / n_tok
                loss = loss_fine + coarse_loss_weight * loss_coarse
            else:
                loss = loss_fine

            # 逆伝播
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            tot_loss += loss.item()
            steps += 1

        avg = tot_loss / max(1, steps)
        print(f"[ep {ep}] loss={avg:.4f}")

        # 疑似 validation: train loss が改善したときだけ保存
        if avg < best - 1e-4:
            best = avg
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))
            print(f"  -> best updated, saved to {ckpt_dir}/best.pt")
        else:
            no_improve += 1
            print(f"  -> no_improve = {no_improve}/{patience}")
            if no_improve >= patience:
                print("[train] early stopping triggered")
                break

    summary = {
        "best_train_loss": best,
        "epochs_trained": ep + 1,
        "seed": seed,
        "device": device,
    }
    summary_path = os.path.join(metrics_dir, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[train] wrote summary to {summary_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
