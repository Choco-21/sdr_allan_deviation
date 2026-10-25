from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from sdr_allan_deviation.readers import (
    iter_iq_chunks,
    load_iq_head,
    print_input_info,
)


class ReaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.iq_int16 = np.array(
            [
                [-32768, 32767],
                [-16384, 8192],
                [0, -1],
                [12345, -23456],
            ],
            dtype="<i2",
        )
        self.expected = (
            self.iq_int16[:, 0].astype(np.float32) / 32768.0
            + 1j * self.iq_int16[:, 1].astype(np.float32) / 32768.0
        ).astype(np.complex64)

        self.dat_path = self.root / "test.dat"
        self.iq_int16.tofile(self.dat_path)

        self.raw_path = self.root / "test.raw"
        raw_frames = np.zeros((self.iq_int16.shape[0], 4), dtype="<i2")
        raw_frames[:, :2] = self.iq_int16
        raw_frames[:, 2:] = 123
        raw_frames.tofile(self.raw_path)

        self.wav_path = self.root / "test.wav"
        sf.write(self.wav_path, self.iq_int16, 20_000_000, subtype="PCM_16")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_all_formats_return_same_normalized_iq(self) -> None:
        for path in (self.dat_path, self.raw_path, self.wav_path):
            with self.subTest(suffix=path.suffix):
                iq, sample_rate = load_iq_head(path, 20_000_000.0, 100)
                np.testing.assert_array_equal(iq, self.expected)
                self.assertEqual(sample_rate, 20_000_000.0)

    def test_all_formats_stream_same_iq(self) -> None:
        for path in (self.dat_path, self.raw_path, self.wav_path):
            with self.subTest(suffix=path.suffix):
                iq = np.concatenate(list(iter_iq_chunks(path, 20_000_000.0)))
                np.testing.assert_array_equal(iq, self.expected)

    def test_binary_head_honors_start_sample(self) -> None:
        for path in (self.dat_path, self.raw_path):
            with self.subTest(suffix=path.suffix):
                iq, _ = load_iq_head(path, 20_000_000.0, 2, start_sample=1)
                np.testing.assert_array_equal(iq, self.expected[1:3])

    def test_incomplete_dat_frame_is_rejected(self) -> None:
        malformed = self.root / "malformed.dat"
        malformed.write_bytes(b"\x00" * 6)
        with self.assertRaisesRegex(ValueError, "trailing byte"):
            list(iter_iq_chunks(malformed, 20_000_000.0))

    def test_dat_input_info_uses_configured_rate(self) -> None:
        with patch("builtins.print") as print_mock:
            print_input_info(self.dat_path, 20_000_000.0)
        output = "\n".join(
            " ".join(str(value) for value in call.args)
            for call in print_mock.call_args_list
        )
        self.assertIn("USRP sc16", output)
        self.assertIn("[I, Q]", output)
        self.assertIn("2e-07 s", output)


if __name__ == "__main__":
    unittest.main()
