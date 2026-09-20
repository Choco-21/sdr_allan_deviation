from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import soundfile as sf

# 1 フレーム = [RX1_I, RX1_Q, RX2_I, RX2_Q]
INTS_PER_FRAME = 4
BYTES_PER_FRAME = INTS_PER_FRAME * 2
CHUNK_FRAMES = 2_000_000


def _raw_frames_to_iq(raw: np.ndarray) -> np.ndarray:
    n_frames = raw.size // INTS_PER_FRAME
    raw = raw[: n_frames * INTS_PER_FRAME]
    i = raw[0::INTS_PER_FRAME].astype(np.float32)
    q = raw[1::INTS_PER_FRAME].astype(np.float32)
    return i + 1j * q


def _iter_raw_iq_chunks(path: Path) -> Iterator[np.ndarray]:
    file_size = path.stat().st_size
    if file_size < BYTES_PER_FRAME:
        raise ValueError("not enough data to form one IQ frame")
    offset = 0
    while offset < file_size:
        raw = np.fromfile(
            path,
            dtype=np.int16,
            count=CHUNK_FRAMES * INTS_PER_FRAME,
            offset=offset,
        )
        if raw.size < INTS_PER_FRAME:
            break
        n_frames = raw.size // INTS_PER_FRAME
        yield _raw_frames_to_iq(raw)
        offset += n_frames * BYTES_PER_FRAME


def _iter_wav_iq_chunks(
    path: Path,
    source_rate_hz: float | None,
) -> Iterator[np.ndarray]:
    try:
        with sf.SoundFile(path) as file:
            if file.channels < 2:
                raise ValueError("IQ input must have at least 2 channels: I and Q")
            while True:
                block = file.read(CHUNK_FRAMES, dtype="float32", always_2d=True)
                if block.size == 0:
                    break
                yield block[:, 0] + 1j * block[:, 1]
    except sf.LibsndfileError:
        if source_rate_hz is None:
            raise ValueError("raw IQ には source_rate_hz の指定が必要です。") from None
        iq, _ = sf.read(
            path,
            dtype="float32",
            always_2d=True,
            samplerate=int(source_rate_hz),
            channels=2,
            format="RAW",
            subtype="PCM_U8",
        )
        if iq.shape[1] < 2:
            raise ValueError("IQ input must have at least 2 channels: I and Q")
        data = iq[:, 0] + 1j * iq[:, 1]
        for start in range(0, data.size, CHUNK_FRAMES):
            yield data[start : start + CHUNK_FRAMES]


def _load_raw_head(
    path: Path,
    source_rate_hz: float,
    num_samples: int,
    start_sample: int = 0,
) -> tuple[np.ndarray, float]:
    raw = np.fromfile(
        path,
        dtype=np.int16,
        count=num_samples * INTS_PER_FRAME,
        offset=start_sample * BYTES_PER_FRAME,
    )
    if raw.size < INTS_PER_FRAME:
        raise ValueError("not enough data to form one IQ frame")
    return _raw_frames_to_iq(raw), float(source_rate_hz)


def _load_wav_head(
    path: Path,
    source_rate_hz: float,
    num_samples: int,
    start_sample: int = 0,
) -> tuple[np.ndarray, float]:
    try:
        iq, fs = sf.read(
            path,
            start=start_sample,
            frames=num_samples,
            dtype="float32",
            always_2d=True,
        )
    except sf.LibsndfileError:
        iq, fs = sf.read(
            path,
            start=start_sample,
            frames=num_samples,
            dtype="float32",
            always_2d=True,
            samplerate=int(source_rate_hz),
            channels=2,
            format="RAW",
            subtype="PCM_U8",
        )
    if iq.shape[1] < 2:
        raise ValueError("IQ input must have at least 2 channels: I and Q")
    if source_rate_hz is not None:
        fs = float(source_rate_hz)
    if iq.size == 0:
        raise ValueError("no IQ samples in wav file")
    return iq[:, 0] + 1j * iq[:, 1], float(fs)


def iter_iq_chunks(path: Path, source_rate_hz: float) -> Iterator[np.ndarray]:
    """拡張子に応じて複素 IQ チャンクを返す。新しい形式はここに足す。"""
    if not path.exists():
        raise FileNotFoundError(f"{path} が見つかりません。")
    suffix = path.suffix.lower()
    if suffix == ".raw":
        yield from _iter_raw_iq_chunks(path)
    elif suffix == ".wav":
        yield from _iter_wav_iq_chunks(path, source_rate_hz)
    else:
        raise ValueError(f"unsupported IQ format: {suffix}")


def load_iq_head(
    path: Path,
    source_rate_hz: float,
    num_samples: int,
    start_sample: int = 0,
) -> tuple[np.ndarray, float]:
    """``start_sample`` から ``num_samples`` 点だけを元レートで読む。"""
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if start_sample < 0:
        raise ValueError("start_sample must be >= 0")
    if not path.exists():
        raise FileNotFoundError(f"{path} が見つかりません。")
    suffix = path.suffix.lower()
    if suffix == ".raw":
        return _load_raw_head(path, source_rate_hz, num_samples, start_sample)
    if suffix == ".wav":
        return _load_wav_head(path, source_rate_hz, num_samples, start_sample)
    raise ValueError(f"unsupported IQ format: {suffix}")
