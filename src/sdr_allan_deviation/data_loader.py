from __future__ import annotations

from pathlib import Path

import numpy as np

from sdr_allan_deviation.filters import decimate_stream
from sdr_allan_deviation.readers import iter_iq_chunks


def load_iq_decimated(
    path: Path,
    source_rate_hz: float,
    target_rate_hz: float,
    cutoff_hz: float,
    highpass_hz: float = 0.0,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """チャンク読み + フィルタのあと、指定レートへ間引く。

    戻り値は ``(iq, actual_rate, raw_freqs, raw_psd)``。
    raw スペクトルは FIR 前のフルレート平均ピリオドグラム。
    """
    chunks = iter_iq_chunks(path, source_rate_hz)
    return decimate_stream(
        chunks,
        source_rate_hz,
        target_rate_hz,
        cutoff_hz,
        highpass_hz=highpass_hz,
    )
