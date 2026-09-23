from __future__ import annotations

from pathlib import Path

import allantools
import numpy as np
from scipy.signal import welch

PERIODGRAM_NFFT = 65_536


class PeriodogramAccumulator:
    """非重畳 Hann ピリオドグラムの平均をチャンク横断で積む。"""

    def __init__(self, sample_rate: float, nfft: int = PERIODGRAM_NFFT) -> None:
        if nfft < 8:
            raise ValueError("nfft must be >= 8")
        self.sample_rate = float(sample_rate)
        self.nfft = int(nfft)
        self.window = np.hanning(self.nfft).astype(np.float64)
        self.window_power = float(np.sum(self.window ** 2))
        self.sum_psd = np.zeros(self.nfft, dtype=np.float64)
        self.n_seg = 0
        self.leftover = np.empty(0, dtype=np.complex64)

    def add(self, iq: np.ndarray) -> None:
        if iq.size == 0:
            return
        if self.leftover.size:
            iq = np.concatenate([self.leftover, np.asarray(iq)])
        n = iq.size // self.nfft * self.nfft
        self.leftover = np.asarray(iq[n:], dtype=np.complex64)
        if n == 0:
            return
        segs = np.asarray(iq[:n], dtype=np.complex128).reshape(-1, self.nfft)
        spec = np.fft.fft(segs * self.window, axis=1)
        self.sum_psd += np.sum(np.abs(spec) ** 2, axis=0) / self.window_power
        self.n_seg += segs.shape[0]

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n_seg < 1:
            raise ValueError("not enough samples for periodogram")
        freqs = np.fft.fftshift(np.fft.fftfreq(self.nfft, d=1.0 / self.sample_rate))
        psd = np.fft.fftshift(self.sum_psd / self.n_seg)
        return freqs, psd


def average_periodogram(
    iq: np.ndarray,
    sample_rate: float,
    nfft: int = PERIODGRAM_NFFT,
) -> tuple[np.ndarray, np.ndarray]:
    acc = PeriodogramAccumulator(sample_rate, nfft)
    acc.add(iq)
    return acc.result()


def spectrum_peak(freqs: np.ndarray, psd: np.ndarray) -> tuple[float, float]:
    peak = int(np.argmax(psd))
    return float(freqs[peak]), float(psd[peak])


def instantaneous_frequency_psd(
    freq_hz: np.ndarray,
    sample_rate: float,
    nfft: int = PERIODGRAM_NFFT,
) -> tuple[np.ndarray, np.ndarray]:
    """平均を除いた瞬時周波数の片側 PSD（Hz²/Hz）。"""
    data = np.asarray(freq_hz, dtype=np.float64)
    if data.size < 8:
        raise ValueError("not enough samples for frequency PSD")
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be > 0")
    nperseg = min(int(nfft), data.size)
    freqs, psd = welch(
        data - np.mean(data),
        fs=sample_rate,
        nperseg=nperseg,
        scaling="density",
    )
    return freqs, psd


def binned_fractional_frequency(
    iq: np.ndarray,
    sample_rate: float,
    reference_hz: float,
    bin_seconds: float,
) -> np.ndarray:
    delta = np.angle(iq[1:] * np.conj(iq[:-1]))
    fractional = delta * sample_rate / (2.0 * np.pi * reference_hz)

    bin_samples = max(1, int(round(bin_seconds * sample_rate)))
    n = fractional.size // bin_samples * bin_samples
    return fractional[:n].reshape(-1, bin_samples).mean(axis=1)


def required_allan_deviation(taus: np.ndarray, observation_frequency: float) -> np.ndarray:
    return 1.0 / (2.0 * np.pi * observation_frequency * taus)


def octave_taus(min_tau: float, max_tau: float) -> list[float]:
    taus: list[float] = []
    tau = min_tau
    while tau <= max_tau:
        taus.append(tau)
        tau *= 2.0
    if not taus or taus[-1] < max_tau:
        taus.append(max_tau)
    return taus


def compute_and_plot_adev(
    data: np.ndarray,
    rate: float,
    plot_path: Path,
    taus: list[float] | None,
    max_tau_seconds: float,
    min_tau_seconds: float,
    observation_frequency: float | None,
    title: str,
) -> None:
    if data.size < 2:
        raise SystemExit("not enough samples to compute Allan deviation")

    tau_list = taus if taus is not None else octave_taus(min_tau_seconds, max_tau_seconds)
    dataset = allantools.Dataset(
        data,
        rate=rate,
        data_type="freq",
        taus=tau_list,
    )
    dataset.compute("adev")

    deviations = np.asarray(dataset.out["stat"], dtype=np.float64)
    if not np.any(np.isfinite(deviations) & (deviations > 0.0)):
        raise SystemExit(
            "正のアラン偏差がないためロググラフを描けません。"
            "入力データが一定値、または変動が小さすぎる可能性があります。"
        )

    plot = allantools.Plot(no_display=True)
    plot.plot(dataset, marker="o", grid=True, label="measured", errorbars=True, capsize=3)

    tau_values = np.asarray(dataset.out["taus"], dtype=np.float64)
    if observation_frequency is not None:
        required = required_allan_deviation(tau_values, observation_frequency)
        plot.ax.loglog(tau_values, required, "--", label="required")

    plot.ax.set_xlim(tau_list[0], max_tau_seconds)
    plot.ax.set_title(title)
    plot.ax.legend()
    plot.fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot.save(plot_path)
    plot.plt.close(plot.fig)

    print(
        f"wrote {len(dataset.out['taus'])} Allan deviation points "
        f"(fractional) to {plot_path}"
    )
