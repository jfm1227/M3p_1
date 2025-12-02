# src/m3p/model/t2mouth.py

import torch
import torch.nn as nn


class TinyEncoder(nn.Module):
    def __init__(
        self,
        vocab: int,
        emb_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.att = nn.Linear(emb_dim, 1)

    def forward(self, x, mask):
        """
        x: [B, L]
        mask: [B, L] (True=valid, False=pad)
        return: [B, emb_dim]
        """
        h = self.emb(x)                              # [B, L, D]
        h = self.enc(h, src_key_padding_mask=~mask)  # [B, L, D]

        s = self.att(h).squeeze(-1)                  # [B, L]
        s = s.masked_fill(~mask, -1e9)
        w = torch.softmax(s, dim=-1).unsqueeze(-1)   # [B, L, 1]

        return (h * w).sum(1)                        # [B, D]


class TinyDecoder(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(emb_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True)

    def forward(self, c, Tlist):
        """
        c: [B, D] (encoder の文ベクトル)
        Tlist: list[int] (各サンプルの長さ)
        return: h_dec [B, maxT, D]
        """
        maxT = max(int(T) for T in Tlist)
        if maxT <= 0:
            # 安全側: 長さ0のときは 1 step だけでも出す
            maxT = 1

        # [B, 1, D] -> [B, maxT, D]
        inp = torch.tanh(self.fc(c)).unsqueeze(1).repeat(1, maxT, 1)
        out, _ = self.gru(inp)  # [B, maxT, D]
        return out


class Text2Mouth(nn.Module):
    """
    マルチタスク版 Text2Mouth

    - fine head: 6クラス (close / a / i / u / e / o)
    - coarse head: 3クラス (close, 横系(i/e), 縦系(a/u/o))
    """

    def __init__(
        self,
        vocab: int,
        emb_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        ncls: int = 6,
        ncls_coarse: int = 3,
        multitask: bool = False,
    ):
        super().__init__()
        self.enc = TinyEncoder(
            vocab=vocab,
            emb_dim=emb_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.dec = TinyDecoder(emb_dim=emb_dim)

        self.head_fine = nn.Linear(emb_dim, ncls)
        self.multitask = multitask
        self.head_coarse = (
            nn.Linear(emb_dim, ncls_coarse) if multitask else None
        )

    def forward(self, x, mask, Tlist, return_coarse: bool = False):
        """
        x: [B, L]
        mask: [B, L]
        Tlist: list[int]
        return:
          - return_coarse=False: logits_fine [B, maxT, 6]
          - return_coarse=True : (logits_fine [B, maxT, 6], logits_coarse [B, maxT, 3])
        """
        c = self.enc(x, mask)          # [B, D]
        h = self.dec(c, Tlist)         # [B, maxT, D]
        logits_fine = self.head_fine(h)  # [B, maxT, 6]

        if self.multitask or return_coarse:
            if self.head_coarse is None:
                raise RuntimeError(
                    "Text2Mouth: multitask=True で初期化していないのに "
                    "return_coarse=True が指定されました。"
                )
            logits_coarse = self.head_coarse(h)  # [B, maxT, 3]
            if return_coarse:
                return logits_fine, logits_coarse

        return logits_fine
