from pathlib import Path
import sys
import unittest

import ephem
import numpy as np

from real_environment.maneuver_truth import (
    apply_rtn_impulse,
    create_post_burn_ephem,
    create_truth_trajectory,
    position_teme_km,
    rtn_basis,
    rv_to_smith_elements,
    smith_elements_to_rv,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SMITH_SOURCE = REPOSITORY_ROOT / "try_code" / "MARLSSA-main"
if str(SMITH_SOURCE) not in sys.path:
    sys.path.insert(0, str(SMITH_SOURCE))

from Environments.SSAMultiUtils import ephem_initial_random


class ManeuverTruthTests(unittest.TestCase):
    def test_rtn_basis_is_right_handed_and_orthonormal(self):
        position = np.array([7000.0, 2000.0, 1000.0])
        velocity = np.array([-1.0, 7.0, 2.0])
        basis = rtn_basis(position, velocity)

        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.cross(basis[:, 0], basis[:, 1]), basis[:, 2], atol=1e-12)

    def test_rtn_impulse_changes_velocity_by_requested_magnitude(self):
        position = np.array([42164.0, 0.0, 0.0])
        velocity = np.array([0.0, 3.0747, 0.0])
        _, post_velocity, delta_v_teme = apply_rtn_impulse(
            position, velocity, np.array([1.0, 5.0, -2.0])
        )

        np.testing.assert_allclose(post_velocity - velocity, delta_v_teme, atol=1e-15)
        self.assertAlmostEqual(np.linalg.norm(delta_v_teme) * 1000.0, np.sqrt(30.0), places=12)

    def test_cartesian_element_round_trip(self):
        elements = np.array([37.0, 112.0, 0.12, 48.0, 215.0, 1.0027])
        position, velocity = smith_elements_to_rv(elements)
        recovered = rv_to_smith_elements(position, velocity)
        recovered_position, recovered_velocity = smith_elements_to_rv(recovered)

        np.testing.assert_allclose(recovered_position, position, atol=1e-8)
        np.testing.assert_allclose(recovered_velocity, velocity, atol=1e-11)

    def test_zero_delta_v_reuses_nominal_truth_object(self):
        np.random.seed(7)
        burn_time = ephem.Date("2020/01/01 19:15:00.00")
        truth_objects, _, _ = ephem_initial_random(1, ephem.Date("2020/01/01 19:00:00.00"))
        post_burn, report = create_post_burn_ephem(truth_objects[0], burn_time, np.zeros(3))

        self.assertIs(post_burn, truth_objects[0])
        self.assertLess(report.pyephem_epoch_position_discontinuity_km, 1e-9)
        self.assertLess(report.reconstruction_position_error_km, 1e-8)
        self.assertLess(report.reconstruction_velocity_error_mps, 1e-8)

    def test_nonzero_delta_v_is_deterministic(self):
        np.random.seed(11)
        burn_time = ephem.Date("2020/01/01 19:15:00.00")
        truth_objects, _, _ = ephem_initial_random(1, ephem.Date("2020/01/01 19:00:00.00"))
        _, report_1 = create_post_burn_ephem(truth_objects[0], burn_time, np.array([0.0, 10.0, 0.0]))
        _, report_2 = create_post_burn_ephem(truth_objects[0], burn_time, np.array([0.0, 10.0, 0.0]))

        np.testing.assert_allclose(report_1.post_burn_elements, report_2.post_burn_elements, atol=0.0)
        np.testing.assert_allclose(report_1.delta_v_teme_mps, report_2.delta_v_teme_mps, atol=0.0)

    def test_zero_delta_v_trajectory_is_exactly_nominal(self):
        np.random.seed(13)
        initial_time = ephem.Date("2020/01/01 19:00:00.00")
        burn_time = ephem.Date(initial_time + 900.0 * ephem.second)
        truth_objects, _, _ = ephem_initial_random(1, initial_time)
        trajectory = create_truth_trajectory(truth_objects[0], burn_time, np.zeros(3))

        for elapsed_seconds in (0.0, 899.0, 900.0, 1800.0, 5400.0):
            evaluation_time = ephem.Date(initial_time + elapsed_seconds * ephem.second)
            np.testing.assert_array_equal(
                trajectory.position_km(evaluation_time),
                position_teme_km(truth_objects[0], evaluation_time),
            )

    def test_cartesian_maneuver_is_continuous_and_applies_exact_impulse(self):
        np.random.seed(17)
        initial_time = ephem.Date("2020/01/01 19:00:00.00")
        burn_time = ephem.Date(initial_time + 900.0 * ephem.second)
        truth_objects, _, _ = ephem_initial_random(1, initial_time)
        requested_delta_v = np.array([2.0, 10.0, -3.0])
        trajectory = create_truth_trajectory(
            truth_objects[0], burn_time, requested_delta_v
        )

        burn_position, post_velocity = trajectory.state_km_s(burn_time)
        np.testing.assert_allclose(
            burn_position,
            position_teme_km(truth_objects[0], burn_time),
            atol=0.0,
        )
        np.testing.assert_allclose(
            post_velocity - trajectory.pre_burn_velocity_km_s,
            trajectory.delta_v_teme_km_s,
            atol=1e-15,
        )
        self.assertAlmostEqual(
            np.linalg.norm(trajectory.delta_v_teme_km_s) * 1000.0,
            np.linalg.norm(requested_delta_v),
            places=12,
        )

    def test_cartesian_post_burn_propagation_is_deterministic(self):
        np.random.seed(19)
        initial_time = ephem.Date("2020/01/01 19:00:00.00")
        burn_time = ephem.Date(initial_time + 900.0 * ephem.second)
        evaluation_time = ephem.Date(initial_time + 3600.0 * ephem.second)
        truth_objects, _, _ = ephem_initial_random(1, initial_time)
        trajectory = create_truth_trajectory(
            truth_objects[0], burn_time, np.array([0.0, 10.0, 0.0])
        )

        position_1, velocity_1 = trajectory.state_km_s(evaluation_time)
        position_2, velocity_2 = trajectory.state_km_s(evaluation_time)
        np.testing.assert_array_equal(position_1, position_2)
        np.testing.assert_array_equal(velocity_1, velocity_2)


if __name__ == "__main__":
    unittest.main()
