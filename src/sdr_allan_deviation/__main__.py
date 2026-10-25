from __future__ import annotations

import argparse
from pathlib import Path

from sdr_allan_deviation.config import (
    BIN_SECONDS,
    HIGHPASS_HZ,
    LOWPASS_HZ,
    MAX_TAU_SECONDS,
    OBSERVATION_FREQUENCY,
    OUTPUTS_DIR,
    REFERENCE_HZ,
    SIGNAL_PLOT_SECONDS,
    SKIP_SECONDS,
    SOURCE_RATE_HZ,
    TARGET_RATE_HZ,
    TAUS,
)
from sdr_allan_deviation.data_loader import load_iq_decimated
from sdr_allan_deviation.filters import apply_fir, design_lowpass_fir, highpass_filter_iq
from sdr_allan_deviation.plots import (
    plot_instantaneous_frequency,
    plot_instantaneous_frequency_psd,
    plot_iq_signal,
    plot_power_spectra,
)
from sdr_allan_deviation.processing import (
    average_periodogram,
    binned_fractional_frequency,
    compute_and_plot_adev,
    instantaneous_frequency_psd,
)
from sdr_allan_deviation.readers import load_iq_head, print_input_info


def parse_args() -> Path:
    parser = argparse.ArgumentParser(description="IQ Allan deviation analysis")
    parser.add_argument("input", type=Path, help="IQ input (.wav, .raw, or .dat)")
    args = parser.parse_args()
    path = args.input.expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"{path} が見つかりません。")
    return path


def main() -> None:
    input_path = parse_args()
    print_input_info(input_path, SOURCE_RATE_HZ)
    out_dir = OUTPUTS_DIR / input_path.name
    out_dir.mkdir(parents=True, exist_ok=True)
    signal_plot_path = out_dir / "signal_iq.png"
    spectrum_plot_path = out_dir / "power_spectrum.png"
    freq_plot_path = out_dir / "instantaneous_frequency.png"
    freq_psd_plot_path = out_dir / "instantaneous_frequency_psd.png"
    adev_plot_path = out_dir / "allan_deviation.png"

    skip_source = max(0, int(round(SKIP_SECONDS * SOURCE_RATE_HZ)))
    head_samples = max(2, int(round(SIGNAL_PLOT_SECONDS * SOURCE_RATE_HZ)))
    iq_head, source_rate = load_iq_head(
        input_path,
        SOURCE_RATE_HZ,
        head_samples,
        start_sample=skip_source,
    )
    taps = design_lowpass_fir(LOWPASS_HZ, source_rate)
    iq_head_filtered, _ = apply_fir(iq_head, taps)
    iq_head_filtered = highpass_filter_iq(iq_head_filtered, source_rate, HIGHPASS_HZ)
    plot_iq_signal(
        iq_head,
        iq_head_filtered,
        source_rate,
        signal_plot_path,
        SIGNAL_PLOT_SECONDS,
        LOWPASS_HZ,
        HIGHPASS_HZ,
    )

    iq, sample_rate, raw_freqs, raw_psd = load_iq_decimated(
        input_path,
        SOURCE_RATE_HZ,
        TARGET_RATE_HZ,
        LOWPASS_HZ,
        highpass_hz=HIGHPASS_HZ,
    )
    skip = max(0, int(round(SKIP_SECONDS * sample_rate)))
    if skip >= iq.size:
        raise SystemExit(
            f"skip {SKIP_SECONDS:g} s がデータ長 {iq.size / sample_rate:g} s 以上です。"
        )
    if skip:
        iq = iq[skip:]
    duration_s = iq.size / sample_rate
    print(f"input: {input_path}")
    print(f"output: {out_dir}")
    print(f"skipped first {SKIP_SECONDS:g} s")
    print(f"duration: {duration_s:g} s ({iq.size} samples)")
    print(
        f"sample_rate={sample_rate:g} Hz "
        f"(FIR LPF {LOWPASS_HZ:g} Hz, HPF {HIGHPASS_HZ:g} Hz, "
        f"source={SOURCE_RATE_HZ:g} Hz)"
    )

    ds_freqs, ds_psd = average_periodogram(iq, sample_rate)
    plot_power_spectra(
        raw_freqs,
        raw_psd,
        ds_freqs,
        ds_psd,
        spectrum_plot_path,
        lowpass_hz=LOWPASS_HZ,
        highpass_hz=HIGHPASS_HZ,
    )

    data = binned_fractional_frequency(iq, sample_rate, REFERENCE_HZ, BIN_SECONDS)
    print(f"fractional-frequency bins: {data.size}, cadence={BIN_SECONDS:g} s")

    freq_hz = data * REFERENCE_HZ
    plot_instantaneous_frequency(freq_hz, BIN_SECONDS, freq_plot_path)
    freq_psd_freqs, freq_psd = instantaneous_frequency_psd(freq_hz, 1.0 / BIN_SECONDS)
    plot_instantaneous_frequency_psd(freq_psd_freqs, freq_psd, freq_psd_plot_path)

    compute_and_plot_adev(
        data,
        1.0 / BIN_SECONDS,
        adev_plot_path,
        TAUS,
        MAX_TAU_SECONDS,
        BIN_SECONDS,
        OBSERVATION_FREQUENCY,
        title="Allan Deviation (IQ, filtered)",
    )


if __name__ == "__main__":
    main()
