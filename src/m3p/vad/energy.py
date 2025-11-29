# src/m3p/vad/energy.py

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import numpy as np
import soundfile as sf


@dataclass
class VadSegment:
    """1 区間分の VAD 結果."""
    type: str        # "speech" or "silence"
    start_ms: int
    end_ms: int


def compute_frame_energy(
    wav_path: str,
    frame_ms: int = 20,
    hop_ms: int = 10,
) -> tuple[np.ndarray, int, int, int, int]:
    """
    WAV を読み込み、フレームごとのエネルギーを計算する。

    Returns
    -------
    energies : np.ndarray  shape = (n_frames,)
    sr       : int         サンプリングレート
    frame_ms : int         使用されたフレーム長 (ms)
    hop_ms   : int         使用されたホップ長 (ms)
    n_samples: int         生波形のサンプル数
    """
    y, sr = sf.read(wav_path)
    # ステレオならモノラルに
    if y.ndim > 1:
        y = y.mean(axis=1)

    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    if frame_len <= 0 or hop_len <= 0:
        raise ValueError("frame_ms / hop_ms が小さすぎます。")

    energies: List[float] = []
    for start in range(0, len(y) - frame_len + 1, hop_len):
        frame = y[start:start + frame_len]
        e = float(np.mean(frame ** 2))
        energies.append(e)

    return np.asarray(energies, dtype=float), sr, frame_ms, hop_ms, len(y)


def energy_vad(
    energies: np.ndarray,
    frame_ms: int = 20,
    hop_ms: int = 10,
    thr: float = 1e-4,
    min_speech_ms: int = 100,
    min_silence_ms: int = 100,
) -> List[VadSegment]:
    """
    エネルギー系列に閾値を当てるだけの簡易 VAD。

    Parameters
    ----------
    energies       : 各フレームのエネルギー
    frame_ms       : フレーム長 (ms)  ※現状ロジックでは hop_ms のみ使用
    hop_ms         : ホップ長 (ms)
    thr            : エネルギーしきい値
    min_speech_ms  : speech 区間として採用する最小長
    min_silence_ms : silence 区間として採用する最小長

    Returns
    -------
    segments : VadSegment のリスト
    """
    flags = energies > thr
    segments: List[VadSegment] = []
    curr: Optional[tuple[str, int, int]] = None  # (kind, start_idx, end_idx)

    def flush_segment(kind: str, start_idx: int, end_idx: int) -> None:
        if start_idx is None:
            return
        dur_ms = (end_idx - start_idx) * hop_ms
        if kind == "speech" and dur_ms < min_speech_ms:
            return
        if kind == "silence" and dur_ms < min_silence_ms:
            return
        seg = VadSegment(
            type=kind,
            start_ms=start_idx * hop_ms,
            end_ms=end_idx * hop_ms,
        )
        segments.append(seg)

    for i, is_speech in enumerate(flags):
        if curr is None:
            curr = ("speech" if is_speech else "silence", i, i + 1)
        else:
            kind, s, e = curr
            if (is_speech and kind == "speech") or ((not is_speech) and kind == "silence"):
                # 同じクラスが続く
                curr = (kind, s, e + 1)
            else:
                # クラスが変わったので 1 区切り
                flush_segment(kind, s, e)
                curr = ("speech" if is_speech else "silence", i, i + 1)

    if curr is not None:
        flush_segment(*curr)

    return segments


def run_energy_vad(
    wav_path: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    設定 dict を受け取り、1 本の wav に対して VAD を実行し、
    JSON にそのまま保存できる dict を返すユーティリティ。

    cfg で主に使うキー:
      - session_id (optional)
      - utt_id     (optional)
      - frame_ms
      - hop_ms
      - energy_thr
      - min_speech_ms
      - min_silence_ms
    """
    frame_ms = int(cfg.get("frame_ms", 20))
    hop_ms = int(cfg.get("hop_ms", 10))
    thr = float(cfg.get("energy_thr", 1e-4))
    min_speech_ms = int(cfg.get("min_speech_ms", 100))
    min_silence_ms = int(cfg.get("min_silence_ms", 100))

    energies, sr, frame_ms, hop_ms, n_samples = compute_frame_energy(
        wav_path,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )
    segments = energy_vad(
        energies,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        thr=thr,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
    )

    audio_ms = int(round(n_samples * 1000 / sr))

    out: Dict[str, Any] = {
        "schema_version": "llm_tts_vad_energy_v0.1",  # VAD 用のミニ版
        "session_id": cfg.get("session_id"),
        "utt_id": cfg.get("utt_id"),
        "wav_path": wav_path,
        "sample_rate": sr,
        "audio_ms": audio_ms,
        "frame_ms": frame_ms,
        "hop_ms": hop_ms,
        "energy_thr": thr,
        "min_speech_ms": min_speech_ms,
        "min_silence_ms": min_silence_ms,
        "segments": [asdict(s) for s in segments],
    }
    return out
