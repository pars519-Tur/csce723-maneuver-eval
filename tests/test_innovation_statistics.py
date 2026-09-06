from pathlib import Path
import sys
import unittest

import numpy as np

from real_environment.innovation_statistics import (
    CHI_SQUARE_3_THRESHOLDS,
    NISRecorder,
    make_update_patch,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SMITH_SOURCE = REPOSITORY_ROOT / "try_code" / "MARLSSA-main"
if str(SMITH_SOURCE) not in sys.path:
    sys.path.insert(0, str(SMITH_SOURCE))

from Environments.SSAMultiUtils import update as smith_update


def make_recorder(rso_ids, maneuvering=(), burn_times=None):
    recorder = NISRecorder(enabled=True)
    recorder.begin_measurement_batch(
        "agent_0", 1800.0, list(rso_ids), set(maneuvering), burn_times or {}
    )
    return recorder


class NISComputationTests(unittest.TestCase):
    def test_nis_matches_the_quadratic_form_of_the_innovation(self):
        recorder = make_recorder([4])
        measurement_covariance = np.diag([4.0, 9.0, 16.0])
        measurement_noise = np.eye(3) * 100.0
        predicted = np.array([[10.0], [20.0], [30.0]])
        measurement = np.array([13.0, 24.0, 35.0])

        recorder.record_update(
            measurement_covariance, measurement_noise, measurement, predicted
        )

        innovation = measurement.reshape(3, 1) - predicted
        expected = float(
            innovation.T
            @ np.linalg.inv(measurement_covariance + measurement_noise)
            @ innovation
        )
        self.assertAlmostEqual(recorder.records[0].nis, expected, places=10)
        self.assertAlmostEqual(
            recorder.records[0].innovation_norm_km, float(np.linalg.norm(innovation))
        )

    def test_records_carry_the_declared_identity_and_burn_timing(self):
        recorder = make_recorder([7, 8], maneuvering=[7], burn_times={7: 1200.0})
        for _ in range(2):
            recorder.record_update(
                np.eye(3), np.eye(3) * 100.0, np.zeros(3), np.zeros((3, 1))
            )
        first, second = recorder.records
        self.assertEqual((first.rso_id, first.is_maneuvering), (7, True))
        self.assertAlmostEqual(first.seconds_after_burn, 600.0)
        self.assertEqual((second.rso_id, second.is_maneuvering), (8, False))
        self.assertTrue(np.isnan(second.seconds_after_burn))

    def test_extra_update_calls_beyond_the_declared_batch_are_ignored(self):
        recorder = make_recorder([1])
        for _ in range(3):
            recorder.record_update(
                np.eye(3), np.eye(3), np.zeros(3), np.zeros((3, 1))
            )
        self.assertEqual(len(recorder.records), 1)

    def test_disabled_recorder_collects_nothing(self):
        recorder = NISRecorder(enabled=False)
        recorder.begin_measurement_batch("agent_0", 0.0, [1], set(), {})
        recorder.record_update(np.eye(3), np.eye(3), np.zeros(3), np.zeros((3, 1)))
        self.assertEqual(recorder.records, [])


class UpdatePatchTests(unittest.TestCase):
    def test_patch_returns_smith_result_unchanged(self):
        rng = np.random.default_rng(0)
        mean = rng.normal(size=(6, 1))
        covariance = np.eye(6) * 0.01
        measurement_covariance = np.eye(3) * 4.0
        measurement_noise = np.eye(3) * 100.0
        cross_correlation = rng.normal(size=(6, 3)) * 0.01
        measurement = rng.normal(size=3) * 5.0
        predicted = rng.normal(size=(3, 1))

        expected_mean, expected_covariance = smith_update(
            mean.copy(), covariance.copy(), measurement_covariance.copy(),
            measurement_noise, cross_correlation, measurement, predicted,
        )

        recorder = make_recorder([0])
        patched = make_update_patch(smith_update, recorder)
        observed_mean, observed_covariance = patched(
            mean.copy(), covariance.copy(), measurement_covariance.copy(),
            measurement_noise, cross_correlation, measurement, predicted,
        )

        np.testing.assert_allclose(observed_mean, expected_mean, rtol=0, atol=0)
        np.testing.assert_allclose(observed_covariance, expected_covariance, rtol=0, atol=0)
        self.assertEqual(len(recorder.records), 1)


class NISDistributionTests(unittest.TestCase):
    def test_consistent_filter_gives_a_chi_square_three_distribution(self):
        """A matched filter must produce NIS with mean 3, which is the whole null test."""
        rng = np.random.default_rng(11)
        measurement_covariance = np.diag([1.0, 2.0, 3.0])
        measurement_noise = np.eye(3) * 100.0
        total = measurement_covariance + measurement_noise
        cholesky = np.linalg.cholesky(total)

        recorder = NISRecorder(enabled=True)
        samples = 20000
        recorder.begin_measurement_batch(
            "agent_0", 0.0, list(range(samples)), set(), {}
        )
        predicted = np.zeros((3, 1))
        for _ in range(samples):
            innovation = cholesky @ rng.normal(size=(3, 1))
            recorder.record_update(
                measurement_covariance, measurement_noise, innovation.reshape(3), predicted
            )

        values = np.array([record.nis for record in recorder.records])
        self.assertAlmostEqual(float(values.mean()), 3.0, delta=0.1)
        self.assertAlmostEqual(
            float(np.mean(values > CHI_SQUARE_3_THRESHOLDS["p95"])), 0.05, delta=0.01
        )
        self.assertAlmostEqual(
            float(np.mean(values > CHI_SQUARE_3_THRESHOLDS["p99"])), 0.01, delta=0.005
        )

    def test_summary_splits_by_maneuver_status_and_burn_timing(self):
        recorder = NISRecorder(enabled=True)
        recorder.begin_measurement_batch(
            "agent_0", 1800.0, [1, 2, 3], {2, 3}, {2: 1200.0, 3: 3000.0}
        )
        for _ in range(3):
            recorder.record_update(
                np.eye(3), np.eye(3) * 100.0, np.ones(3) * 30.0, np.zeros((3, 1))
            )
        summary = recorder.summary()
        self.assertEqual(summary["measurements"], 3)
        self.assertEqual(summary["non_maneuvering"]["measurements"], 1)
        self.assertEqual(summary["maneuvering_after_burn"]["measurements"], 1)
        self.assertEqual(summary["maneuvering_before_burn"]["measurements"], 1)
        self.assertEqual(summary["expected_mean_under_consistency"], 3.0)


if __name__ == "__main__":
    unittest.main()
