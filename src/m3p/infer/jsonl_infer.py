import argparse, yaml, json, torch, time, unicodedata, os
from m3p.data.dataset import VOCAB
from m3p.model.t2mouth import Text2Mouth

MOUTHS = ["close","a","i","u","e","o"]

def to_kana(s):
    try:
        from pykakasi import kakasi
        _k = kakasi()
        _k.setMode("J", "H")
        _k.setMode("K", "H")
        _k.setMode("H", "H")
        conv = _k.getConverter()
        return conv.do(s.strip())
    except:
        return unicodedata.normalize("NFKC", s)

C2I = {c: i + 1 for i, c in enumerate(
    list("ぁあぃいぅうぇえぉおかがきぎくぐけげこご"
         "さざしじすずせぜそぞただちぢっつづてでとど"
         "なにぬねのはばぱひびぴふぶぷへべぺほぼぽ"
         "まみむめもやゃゆゅよょらりるれろわゐゑをんー、。！？.!?\n 　"))
}

def text_to_ids(s, max_len=160):
    ids = [C2I.get(c, 0) for c in s[:max_len]]
    return ids, len(ids)

def main(cfg, in_path, out_path):
    dev = cfg["device"] if torch.cuda.is_available() else "cpu"
    step_ms = cfg["step_ms"]
    out_dir = cfg["paths"]["out_dir"]

    m = Text2Mouth(VOCAB, **cfg["model"]).to(dev)
    m.load_state_dict(torch.load(f"{out_dir}/ckpt/best.pt", map_location=dev))
    m.eval()

    # out_path のディレクトリを自動作成（out/day4 など）
    out_dirname = os.path.dirname(out_path)
    if out_dirname:
        os.makedirs(out_dirname, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            if not line.strip():
                continue

            js = json.loads(line)

            kana = js["kana"]
            T = js["T"]

            x_ids, L = text_to_ids(kana)
            x = torch.tensor([x_ids]).long().to(dev)
            mask = torch.zeros_like(x).bool()
            mask[0, :L] = 1

            with torch.no_grad():
                logits = m(x, mask, [T]).squeeze(0)
                pred_ids = logits.argmax(-1).cpu().tolist()

            # ★ 既存の mouth 系フィールドはすべて削除してから書き込む
            for k in list(js.keys()):
                if k in ["mouth_steps", "mouth_steps_gt", "mouth_timeline", "mouth"]:
                    del js[k]

            # 推論結果のみを mouth_steps_pred として追加
            js["mouth_steps_pred"] = pred_ids

            f_out.write(json.dumps(js, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--in", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, "r", encoding="utf-8"))
    main(cfg, getattr(a, "in"), a.out)
