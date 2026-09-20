from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sdr_allan_deviation.processing import spectrum_peak


def _thin(values: np.ndarray, max_points: int = 200_000) -> tuple[np.ndarray, int]:
    n = values.size
    if n <= max_points:
        return values, 1
    step = int(np.ceil(n / max_points))
    return values[::step], step


def plot_iq_signal(
    iq_raw: np.ndarray,
    iq_filtered: np.ndarray,
    sample_rate: float,
    plot_path: Path,
    plot_seconds: float,
    lowpass_hz: float,
    highpass_hz: float = 0.0,
) -> None:
    n = min(iq_raw.size, max(2, int(round(plot_seconds * sample_rate))))
    time_s = np.arange(n, dtype=np.float64) / sample_rate
    raw = iq_raw[:n]
    filtered = iq_filtered[:n]

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(time_s, raw.real, linewidth=0.6, label="I")
    axes[0].plot(time_s, raw.imag, linewidth=0.6, label="Q")
    axes[0].set_ylabel("amplitude")
    axes[0].set_title("Raw IQ")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_s, filtered.real, linewidth=0.6, label="I")
    axes[1].plot(time_s, filtered.imag, linewidth=0.6, label="Q")
    axes[1].set_ylabel("amplitude")
    axes[1].set_xlabel("time [s]")
    if highpass_hz > 0.0:
        filtered_title = (
            f"Filtered IQ (LPF {lowpass_hz:g} Hz, HPF {highpass_hz:g} Hz)"
        )
    else:
        filtered_title = f"Filtered IQ (LPF {lowpass_hz:g} Hz)"
    axes[1].set_title(filtered_title)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"first {plot_seconds:g} s ({n} samples)")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"wrote signal plot to {plot_path}")


def plot_instantaneous_frequency(
    freq_hz: np.ndarray,
    dt_seconds: float,
    plot_path: Path,
    max_plot_points: int = 200_000,
) -> None:
    if freq_hz.size < 1:
        raise SystemExit("not enough samples to plot instantaneous frequency")

    time_s = (np.arange(freq_hz.size, dtype=np.float64) + 0.5) * dt_seconds
    time_plot, step = _thin(time_s, max_plot_points)
    if step == 1:
        freq_plot = freq_hz
    else:
        freq_plot = freq_hz[::step]

    mean_hz = float(np.mean(freq_hz))
    std_hz = float(np.std(freq_hz))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_plot, freq_plot, linewidth=0.6)
    ax.axhline(mean_hz, color="tab:orange", linestyle="--", linewidth=0.8, label="mean")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("instantaneous frequency [Hz]")
    ax.set_title(f"Instantaneous frequency ({freq_hz.size} samples)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(
        f"instantaneous frequency: mean={mean_hz:g} Hz, std={std_hz:g} Hz "
        f"({freq_hz.size} samples, plot step={step})"
    )
    print(f"wrote instantaneous frequency plot to {plot_path}")


def _plot_spectrum_axis(
    ax: plt.Axes,
    freqs: np.ndarray,
    psd: np.ndarray,
    title: str,
    lowpass_hz: float | None = None,
    highpass_hz: float | None = None,
) -> tuple[float, float]:
    psd_db = 10.0 * np.log10(np.maximum(psd, np.max(psd) * 1e-20))
    peak_hz, peak_psd = spectrum_peak(freqs, psd)
    peak_db = 10.0 * np.log10(max(peak_psd, float(np.max(psd)) * 1e-20))

    ax.plot(freqs, psd_db, linewidth=0.7)
    ax.axvline(peak_hz, color="tab:orange", linestyle="--", linewidth=0.8)
    ax.plot(peak_hz, peak_db, "o", color="tab:orange", markersize=5, label="peak")
    ax.annotate(
        f"{peak_hz:g} Hz",
        xy=(peak_hz, peak_db),
        xytext=(8, 8),
        textcoords="offset points",
        color="tab:orange",
    )
    if lowpass_hz is not None and lowpass_hz > 0.0:
        ax.axvline(-lowpass_hz, color="tab:green", linestyle=":", linewidth=1.0)
        ax.axvline(
            lowpass_hz,
            color="tab:green",
            linestyle=":",
            linewidth=1.0,
            label=f"LPF {lowpass_hz:g} Hz",
        )
    if highpass_hz is not None and highpass_hz > 0.0:
        ax.axvline(-highpass_hz, color="tab:red", linestyle=":", linewidth=1.0)
        ax.axvline(
            highpass_hz,
            color="tab:red",
            linestyle=":",
            linewidth=1.0,
            label=f"HPF {highpass_hz:g} Hz",
        )
    ax.set_ylabel("power [dB]")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    return peak_hz, peak_db


def plot_power_spectra(
    raw_freqs: np.ndarray,
    raw_psd: np.ndarray,
    ds_freqs: np.ndarray,
    ds_psd: np.ndarray,
    plot_path: Path,
    lowpass_hz: float | None = None,
    highpass_hz: float | None = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    raw_peak, _ = _plot_spectrum_axis(
        axes[0],
        raw_freqs,
        raw_psd,
        "Raw IQ (full rate)",
        lowpass_hz=lowpass_hz,
        highpass_hz=highpass_hz,
    )
    ds_peak, _ = _plot_spectrum_axis(
        axes[1],
        ds_freqs,
        ds_psd,
        "After filter + downsample",
        lowpass_hz=lowpass_hz,
        highpass_hz=highpass_hz,
    )
    axes[1].set_xlabel("frequency [Hz]")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"spectrum peak (raw): {raw_peak:g} Hz")
    print(f"spectrum peak (after filter+downsample): {ds_peak:g} Hz")
    print(f"wrote power spectrum plot to {plot_path}")
