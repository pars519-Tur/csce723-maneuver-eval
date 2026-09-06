from pathlib import Path
import sys
import unittest

import ephem
import numpy as np

from real_environment.maneuver_truth import position_teme_km
from real_environment.truth_geometry import ObserverSite, smith_fov_mask, topocentric_az_alt_deg


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SMITH_SOURCE = REPOSITORY_ROOT / "try_code" / "MARLSSA-main"
if str(SMITH_SOURCE) not in sys.path:
    sys.path.insert(0, str(SMITH_SOURCE))

from Environments.SSAMultiUtils import ephem_initial_random


class TruthGeometryTests(unittest.TestCase):
    def test_topocentric_angles_match_pyephem_nominal_truth(self):
        sites = (
            ObserverSite("agent_0", -106.6599, 33.8172, 1250.0),
            ObserverSite("agent_1", -71.488247, 42.623311, 131.0),
            ObserverSite("agent_2", 114.1656095, -21.815978, 5.0),
        )
        initial_time = ephem.Date("2020/01/01 19:00:00.00")
        errors = []
        for seed in range(5):
            np.random.seed(seed)
            truth_objects, _, _ = ephem_initial_random(1, initial_time)
            satellite = truth_objects[0]
            for elapsed_seconds in (0.0, 1800.0, 5400.0):
                time = ephem.Date(initial_time + elapsed_seconds * ephem.second)
                position = position_teme_km(satellite, time)
                for site in sites:
                    observer = site.pyephem_observer(time)
                    satellite.compute(observer)
                    expected_azimuth = math_degrees(satellite.az)
                    expected_altitude = math_degrees(satellite.alt)
                    actual_azimuth, actual_altitude = topocentric_az_alt_deg(position, site, time)
                    azimuth_error = abs((actual_azimuth - expected_azimuth + 180.0) % 360.0 - 180.0)
                    if expected_altitude > 14.0:
                        errors.append((azimuth_error, abs(actual_altitude - expected_altitude)))

        self.assertTrue(errors)
        self.assertLess(max(error[0] for error in errors), 0.05)
        self.assertLess(max(error[1] for error in errors), 0.05)

    def test_smith_fov_mask_selects_cell_center_and_rejects_outside(self):
        mask = smith_fov_mask(
            azimuth_deg=np.array([100.0, 103.0, 100.0, 100.0]),
            altitude_deg=np.array([40.0, 40.0, 43.0, 10.0]),
            past_pointing_azimuth_deg=100.0,
            azimuth_action=44,
            altitude_action=6,
        )
        np.testing.assert_array_equal(mask, np.array([True, False, False, False]))


def math_degrees(angle_radians) -> float:
    return float(angle_radians) * 180.0 / np.pi


if __name__ == "__main__":
    unittest.main()
