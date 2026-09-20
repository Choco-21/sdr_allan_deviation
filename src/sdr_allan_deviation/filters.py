from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.signal import butter, firwin, lfilter, sosfiltfilt

from sdr_allan_deviation.processing import PeriodogramAccumulator

MAX_RATIO_ERROR = 1e-3
FIR_TAPS = 64


def downsample_factor(source_rate_hz: float, target_rate_hz: float) -> tuple[int, float]:
    """整数倍の間引き用 factor と、実際のレートを返す。"""
    if source_rate_hz <= 0.0:
        raise ValueError("source_rate_hz must be > 0")
    if target_rate_hz <= 0.0:
        raise ValueError("target_rate_hz must be > 0")
    if target_rate_hz > source_rate_hz:
        raise ValueError(
            f"target_rate {target_rate_hz:g} Hz must be <= source_rate {source_rate_hz:g} Hz"
        )

    ratio = source_rate_hz / target_rate_hz
    factor = max(1, int(round(ratio)))
    actual_rate = source_rate_hz / factor
    if abs(ratio - factor) / ratio > MAX_RATIO_ERROR:
        raise ValueError(
            f"source/target ratio {ratio:g} is not close to an integer "
            f"(source={source_rate_hz:g} Hz, target={target_rate_hz:g} Hz)"
        )
    return factor, actual_rate


def design_lowpass_fir(
    cutoff_hz: float,
    sample_rate: float,
    numtaps: int = FIR_TAPS,
) -> np.ndarray:
    if cutoff_hz <= 0.0:
        raise ValueError("lowpass cutoff must be > 0")
    nyquist = sample_rate / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"lowpass cutoff {cutoff_hz:g} Hz must be < Nyquist {nyquist:g} Hz"
        )
    return firwin(numtaps, cutoff_hz, fs=sample_rate).astype(np.float32)


def apply_fir(
    iq: np.ndarray,
    taps: np.ndarray,
    zi: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    n_zi = taps.size - 1
    if zi is None:
        zi_i = np.zeros(n_zi, dtype=np.float32)
        zi_q = np.zeros(n_zi, dtype=np.float32)
    else:
        zi_i, zi_q = zi
    i, zi_i = lfilter(taps, 1.0, np.asarray(iq.real, dtype=np.float32), zi=zi_i)
    q, zi_q = lfilter(taps, 1.0, np.asarray(iq.imag, dtype=np.float32), zi=zi_q)
    return (i + 1j * q).astype(np.complex64, copy=False), (zi_i, zi_q)


def highpass_filter_iq(
    iq: np.ndarray,
    sample_rate: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """複素 IQ の I/Q それぞれにゼロ位相 HPF をかける。"""
    if cutoff_hz <= 0.0:
        return np.asarray(iq)
    nyquist = sample_rate / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"highpass cutoff {cutoff_hz:g} Hz must be < Nyquist {nyquist:g} Hz"
        )
    sos = butter(order, cutoff_hz, btype="high", fs=sample_rate, output="sos")
    i = sosfiltfilt(sos, np.asarray(iq.real, dtype=np.float64))
    q = sosfiltfilt(sos, np.asarray(iq.imag, dtype=np.float64))
    return (i + 1j * q).astype(np.complex64)


def _stride_downsample(
    iq: np.ndarray,
    factor: int,
    leftover: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if leftover.size:
        iq = np.concatenate([leftover, iq])
    n = iq.size // factor * factor
    leftover_out = iq[n:]
    if n == 0:
        return iq[:0], leftover_out
    return iq[:n:factor], leftover_out


def decimate_stream(
    chunks: Iterable[np.ndarray],
    source_rate_hz: float,
    target_rate_hz: float,
    cutoff_hz: float,
    highpass_hz: float = 0.0,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """IQ チャンクにローパスと、必要ならハイパスをかけて指定レートへ間引く。"""
    factor, actual_rate = downsample_factor(source_rate_hz, target_rate_hz)
    taps = design_lowpass_fir(cutoff_hz, source_rate_hz)
    zi: tuple[np.ndarray, np.ndarray] | None = None
    leftover = np.empty(0, dtype=np.complex64)
    out: list[np.ndarray] = []
    psd_acc = PeriodogramAccumulator(source_rate_hz)

    for iq in chunks:
        if iq.size == 0:
            continue
        psd_acc.add(iq)
        filtered, zi = apply_fir(iq, taps, zi)
        decimated, leftover = _stride_downsample(filtered, factor, leftover)
        if decimated.size:
            out.append(decimated)

    if leftover.size >= factor:
        decimated, leftover = _stride_downsample(
            leftover,
            factor,
            np.empty(0, dtype=np.complex64),
        )
        if decimated.size:
            out.append(decimated)

    if not out:
        raise ValueError("not enough samples to downsample")
    raw_freqs, raw_psd = psd_acc.result()
    iq = np.concatenate(out)
    if highpass_hz > 0.0:
        iq = highpass_filter_iq(iq, actual_rate, highpass_hz)
    return iq, actual_rate, raw_freqs, raw_psd
