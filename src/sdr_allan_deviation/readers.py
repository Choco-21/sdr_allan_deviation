from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

CHUNK_FRAMES = 2_000_000
INT16_SCALE = 32_768.0


@dataclass(frozen=True)
class InterleavedIQFormat:
    name: str
    dtype: np.dtype
    values_per_frame: int
    i_index: int
    q_index: int
    layout: str

    @property
    def bytes_per_frame(self) -> int:
        return self.dtype.itemsize * self.values_per_frame


BINARY_IQ_FORMATS = {
    ".raw": InterleavedIQFormat(
        name="legacy dual-channel int16",
        dtype=np.dtype("<i2"),
        values_per_frame=4,
        i_index=0,
        q_index=1,
        layout="[RX1_I, RX1_Q, RX2_I, RX2_Q]",
    ),
    ".dat": InterleavedIQFormat(
        name="USRP sc16",
        dtype=np.dtype("<i2"),
        values_per_frame=2,
        i_index=0,
        q_index=1,
        layout="[I, Q]",
    ),
}
SUPPORTED_SUFFIXES = (*BINARY_IQ_FORMATS, ".wav")


def _binary_format(path: Path) -> InterleavedIQFormat:
    try:
        return BINARY_IQ_FORMATS[path.suffix.lower()]
    except KeyError:
        raise ValueError(f"unsupported binary IQ format: {path.suffix.lower()}") from None


def _validate_binary_size(path: Path, iq_format: InterleavedIQFormat) -> int:
    file_size = path.stat().st_size
    if file_size < iq_format.bytes_per_frame:
        raise ValueError("not enough data to form one IQ frame")
    remainder = file_size % iq_format.bytes_per_frame
    if remainder:
        raise ValueError(
            f"{path} has {remainder} trailing byte(s); "
            f"{iq_format.name} requires {iq_format.bytes_per_frame} bytes per frame"
        )
    return file_size // iq_format.bytes_per_frame


def _interleaved_frames_to_iq(
    raw: np.ndarray,
    iq_format: InterleavedIQFormat,
) -> np.ndarray:
    if raw.size % iq_format.values_per_frame:
        raise ValueError(f"incomplete {iq_format.name} IQ frame")
    frames = raw.reshape(-1, iq_format.values_per_frame)
    i = frames[:, iq_format.i_index].astype(np.float32) / INT16_SCALE
    q = frames[:, iq_format.q_index].astype(np.float32) / INT16_SCALE
    return (i + 1j * q).astype(np.complex64, copy=False)


def _iter_binary_iq_chunks(path: Path) -> Iterator[np.ndarray]:
    iq_format = _binary_format(path)
    _validate_binary_size(path, iq_format)
    with path.open("rb") as file:
        while True:
            raw = np.fromfile(
                file,
                dtype=iq_format.dtype,
                count=CHUNK_FRAMES * iq_format.values_per_frame,
            )
            if raw.size == 0:
                break
            yield _interleaved_frames_to_iq(raw, iq_format)


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


def _load_binary_head(
    path: Path,
    source_rate_hz: float,
    num_samples: int,
    start_sample: int = 0,
) -> tuple[np.ndarray, float]:
    iq_format = _binary_format(path)
    _validate_binary_size(path, iq_format)
    raw = np.fromfile(
        path,
        dtype=iq_format.dtype,
        count=num_samples * iq_format.values_per_frame,
        offset=start_sample * iq_format.bytes_per_frame,
    )
    if raw.size < iq_format.values_per_frame:
        raise ValueError("not enough data to form one IQ frame")
    return _interleaved_frames_to_iq(raw, iq_format), float(source_rate_hz)


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


def print_input_info(path: Path, source_rate_hz: float) -> None:
    """入力形式、レート、長さなどを標準出力に出す。"""
    suffix = path.suffix.lower()
    if suffix in BINARY_IQ_FORMATS:
        iq_format = _binary_format(path)
        frames = _validate_binary_size(path, iq_format)
        duration_s = frames / source_rate_hz
        print(f"input info: {path}")
        print(f"  format: {iq_format.name}")
        print(f"  dtype: {iq_format.dtype.str}")
        print(f"  layout: {iq_format.layout}")
        print(f"  samplerate: {source_rate_hz:g} Hz (from config)")
        print(f"  frames: {frames}")
        print(f"  duration: {duration_s:g} s")
        print(f"  file size: {path.stat().st_size} bytes")
        return
    if suffix != ".wav":
        raise ValueError(f"unsupported IQ format: {suffix}")
    try:
        with sf.SoundFile(path) as file:
            duration_s = file.frames / file.samplerate if file.samplerate else 0.0
            print(f"input info: {path}")
            print(f"  format: {file.format} ({file.format_info})")
            print(f"  subtype: {file.subtype} ({file.subtype_info})")
            print(f"  endian: {file.endian}")
            print(f"  samplerate: {file.samplerate} Hz")
            if file.samplerate != source_rate_hz:
                print(f"  analysis samplerate: {source_rate_hz:g} Hz (from config)")
            print(f"  channels: {file.channels}")
            print(f"  frames: {file.frames}")
            print(f"  duration: {duration_s:g} s")
            extra = (file.extra_info or "").strip()
            if extra:
                print("  extra_info:")
                for line in extra.splitlines():
                    print(f"    {line}")
    except sf.LibsndfileError as exc:
        print(f"input info: {path}")
        print(f"  (wav として読めません: {exc})")


def iter_iq_chunks(path: Path, source_rate_hz: float) -> Iterator[np.ndarray]:
    """拡張子に応じて複素 IQ チャンクを返す。新しい形式はここに足す。"""
    if not path.exists():
        raise FileNotFoundError(f"{path} が見つかりません。")
    suffix = path.suffix.lower()
    if suffix in BINARY_IQ_FORMATS:
        yield from _iter_binary_iq_chunks(path)
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
    if suffix in BINARY_IQ_FORMATS:
        return _load_binary_head(path, source_rate_hz, num_samples, start_sample)
    if suffix == ".wav":
        return _load_wav_head(path, source_rate_hz, num_samples, start_sample)
    raise ValueError(f"unsupported IQ format: {suffix}")
