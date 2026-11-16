from __future__ import annotations

import unittest
import zlib

from tandav.kalman import ConstantVelocityFilter
from tandav.protocol import ProtocolError, decode_payload, encode_payload
from tandav.voting import MajorityVote, NVote


class ProtocolTests(unittest.TestCase):
    def make(self):
        return encode_payload(
            seq=4092,
            boot_id="a1b2",
            t_mono_us=12_500_450,
            page="MAIN_TELEMETRY",
            speed=42.5,
            rpm=6800,
            conf=0.98,
            valid=True,
        )

    def test_roundtrip(self):
        obj = decode_payload(self.make())
        self.assertEqual(obj["seq"], 4092)
        self.assertEqual(obj["spd"], 42.5)
        self.assertEqual(obj["rpm"], 6800)
        self.assertTrue(obj["valid"])

    def test_crc_covers_transmitted_prefix(self):
        data = self.make()
        marker = b',"crc":'
        idx = data.rfind(marker)
        body = data[:idx] + b"}"
        sent = int(data[idx + len(marker) : -1])
        self.assertEqual(zlib.crc32(body) & 0xFFFFFFFF, sent)

    def test_tamper_rejected(self):
        data = bytearray(self.make())
        pos = data.find(b"42.5")
        data[pos : pos + 4] = b"92.5"
        with self.assertRaises(ProtocolError):
            decode_payload(bytes(data))

    def test_bad_range_rejected(self):
        with self.assertRaises(ValueError):
            encode_payload(
                seq=1,
                boot_id="aa",
                t_mono_us=1,
                page="MAIN_TELEMETRY",
                speed=-1,
                rpm=6800,
                conf=0.9,
                valid=True,
            )


class VotingTests(unittest.TestCase):
    def test_nvote_requires_consecutive_agreement(self):
        vote = NVote(need=3, window=5)
        self.assertEqual(vote.push(42), (None, False))
        self.assertEqual(vote.push(42), (None, False))
        self.assertEqual(vote.push(42), (42, True))
        self.assertEqual(vote.push(42), (42, False))
        self.assertEqual(vote.push(99), (42, False))
        self.assertEqual(vote.push(99), (42, False))
        self.assertEqual(vote.push(99), (99, True))

    def test_nvote_dropout_breaks_streak(self):
        vote = NVote(need=3, window=5)
        for value in (10, 10, None, 10, 10):
            vote.push(value)
        self.assertIsNone(vote.last)

    def test_majority_page_tolerates_one_bad_frame(self):
        vote = MajorityVote(window=3, need=2)
        self.assertIsNone(vote.push("MAIN"))
        self.assertEqual(vote.push("MAIN"), "MAIN")
        self.assertEqual(vote.push("FUEL"), "MAIN")
        self.assertEqual(vote.push("FUEL"), "FUEL")


class KalmanTests(unittest.TestCase):
    def test_physical_impossibility_rejected(self):
        filt = ConstantVelocityFilter(
            max_accel_per_s=17.0,
            resolution=1.0,
            reject_to_reset=5,
            lo=0,
            hi=120,
        )
        filt.reset(15.0, 0.0)
        accepted = filt.update(16.0, 0.25)
        self.assertAlmostEqual(accepted, 16.0, delta=1.0)
        rejected = filt.update(88.0, 0.50)
        self.assertLess(rejected, 25.0)
        self.assertGreater(filt.rejects, 0)

    def test_normal_track_follows_lcd(self):
        filt = ConstantVelocityFilter(
            max_accel_per_s=17.0,
            resolution=1.0,
            reject_to_reset=5,
            lo=0,
            hi=120,
        )
        filt.reset(30.0, 0.0)
        value = 30.0
        for i in range(1, 20):
            value = 30.0 + i
            out = filt.update(value, i * 0.25)
            self.assertAlmostEqual(out, value, delta=0.51)

    def test_reset_after_persistent_rejection(self):
        filt = ConstantVelocityFilter(
            max_accel_per_s=17.0,
            resolution=1.0,
            reject_to_reset=3,
            lo=0,
            hi=120,
        )
        filt.reset(10.0, 0.0)
        for i in range(1, 4):
            filt.update(90.0, i * 0.25)
        self.assertGreaterEqual(filt.resets, 1)
        self.assertAlmostEqual(filt.value, 90.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
