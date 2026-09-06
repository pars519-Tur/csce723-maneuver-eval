from pathlib import Path
import sys
import unittest

import ephem
import numpy as np

from real_environment.measurement_noise import (
    MeasurementNoiseConfiguration,
    NoisyMeasurementAdapter,
    rso_noise_generator,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SMITH_SOURCE = REPOSITORY_ROOT / "try_code" / "MARLSSA-main"
if str(SMITH_SOURCE) not in sys.path:
    sys.path.insert(0, str(SMITH_SOURCE))

from Environments.SSAMultiUtils import measfun_gt


class ConstantTrajectory:
    def __init__(self, position_km):
        self._position_km = np.asarray(position_km, dtype=float)
        self.call_count = 0

    def position_km(self, time):
        self.call_count += 1
        return self._position_km.copy()


TEST_TIME = ephem.Date("2020/01/01 19:30:00.00")
TEST_POSITION_KM = np.array([32000.0, -21000.0, 4500.0])


class MeasurementNoiseConfigurationTests(unittest.TestCase):
    def test_default_configuration_is_disabled(self):
        self.assertFalse(MeasurementNoiseConfiguration().enabled)

    def test_positive_sigma_is_enabled(self):
        self.assertTrue(MeasurementNoiseConfiguration(sigma_km=10.0).enabled)

    def test_negative_sigma_is_rejected(self):
        with self.assertRaises(ValueError):
            MeasurementNoiseConfiguration(sigma_km=-1.0)


class NoisyMeasurementAdapterTests(unittest.TestCase):
    def test_zero_sigma_round_trips_through_smith_measfun_gt(self):
        """With no noise the adapter must be numerically invisible to Smith's UKF."""
        adapter = NoisyMeasurementAdapter(
            ConstantTrajectory(TEST_POSITION_KM), sigma_km=0.0, generator=None
        )
        measured = np.asarray(measfun_gt(adapter, TEST_TIME), dtype=float).reshape(3)
        np.testing.assert_allclose(measured, TEST_POSITION_KM, rtol=1e-12, atol=1e-9)

    def test_noise_perturbs_the_measurement_at_the_requested_scale(self):
        trajectory = ConstantTrajectory(TEST_POSITION_KM)
        adapter = NoisyMeasurementAdapter(
            trajectory, sigma_km=10.0, generator=np.random.default_rng(0)
        )
        samples = np.array(
            [
                np.asarray(measfun_gt(adapter, TEST_TIME), dtype=float).reshape(3)
                for _ in range(4000)
            ]
        )
        residuals = samples - TEST_POSITION_KM
        np.testing.assert_allclose(residuals.mean(axis=0), np.zeros(3), atol=0.6)
        np.testing.assert_allclose(residuals.std(axis=0), np.full(3, 10.0), rtol=0.06)

    def test_noise_is_recorded_for_audit(self):
        adapter = NoisyMeasurementAdapter(
            ConstantTrajectory(TEST_POSITION_KM),
            sigma_km=10.0,
            generator=np.random.default_rng(3),
        )
        measfun_gt(adapter, TEST_TIME)
        measfun_gt(adapter, TEST_TIME)
        self.assertEqual(len(adapter.applied_noise_km), 2)

    def test_positive_sigma_requires_a_generator(self):
        with self.assertRaises(ValueError):
            NoisyMeasurementAdapter(
                ConstantTrajectory(TEST_POSITION_KM), sigma_km=10.0, generator=None
            )


class NoiseGeneratorIsolationTests(unittest.TestCase):
    def test_generator_does_not_consume_smith_global_numpy_stream(self):
        """Smith's reset draws the catalog from the global stream, so it must stay untouched."""
        configuration = MeasurementNoiseConfiguration(sigma_km=10.0, seed=7)
        np.random.seed(1234)
        expected = np.random.rand(5)

        np.random.seed(1234)
        for rso_id in range(20):
            rso_noise_generator(configuration, 0, rso_id).normal(size=100)
        observed = np.random.rand(5)

        np.testing.assert_array_equal(observed, expected)

    def test_streams_are_reproducible_and_independent_across_rsos_and_seeds(self):
        configuration = MeasurementNoiseConfiguration(sigma_km=10.0, seed=7)
        first = rso_noise_generator(configuration, 3, 11).normal(size=8)
        repeat = rso_noise_generator(configuration, 3, 11).normal(size=8)
        other_rso = rso_noise_generator(configuration, 3, 12).normal(size=8)
        other_seed = rso_noise_generator(configuration, 4, 11).normal(size=8)

        np.testing.assert_array_equal(first, repeat)
        self.assertFalse(np.allclose(first, other_rso))
        self.assertFalse(np.allclose(first, other_seed))


if __name__ == "__main__":
    unittest.main()
