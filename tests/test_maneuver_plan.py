import math
import unittest

import numpy as np

from real_environment.maneuver_plan import (
    EPISODE_LENGTH_SECONDS,
    ExplicitManeuverConfiguration,
    ManeuverConfiguration,
    ManeuverEvent,
    ManeuverPlan,
    RandomManeuverConfiguration,
    plan_from_manifest_events,
    select_maneuvering_rso_ids,
)


class ManeuverPlanTests(unittest.TestCase):
    def test_duplicate_rso_events_are_rejected(self):
        event = ManeuverEvent(rso_id=3, time_seconds=900.0, delta_v_rtn_mps=(0.0, 10.0, 0.0))
        with self.assertRaises(ValueError):
            ManeuverPlan(events=(event, event))

    def test_lookup_helpers_return_the_matching_event(self):
        plan = ManeuverPlan(
            events=(
                ManeuverEvent(7, 1200.0, (1.0, 2.0, 2.0)),
                ManeuverEvent(2, 3000.0, (0.0, 5.0, 0.0)),
            )
        )
        np.testing.assert_array_equal(plan.rso_ids, np.array([2, 7]))
        self.assertEqual(plan.burn_time_for(7), 1200.0)
        np.testing.assert_allclose(plan.delta_v_for(2), np.array([0.0, 5.0, 0.0]))
        self.assertAlmostEqual(plan.event_for(7).magnitude_mps, 3.0)
        with self.assertRaises(KeyError):
            plan.event_for(99)

    def test_manifest_round_trip(self):
        events = [
            {"rso_id": 4, "time_seconds": 1500.5, "delta_v_mps": [1.0, -2.0, 3.0]},
            {"rso_id": 9, "time_seconds": 2500.0, "delta_v_mps": [0.0, 7.0, 0.0]},
        ]
        self.assertEqual(plan_from_manifest_events(events).to_dicts(), events)


class FixedConfigurationTests(unittest.TestCase):
    def test_plan_reproduces_the_original_shared_burn_behaviour(self):
        configuration = ManeuverConfiguration(maneuvering_fraction=0.10)
        plan = configuration.build_plan(100, 0)
        expected_ids = select_maneuvering_rso_ids(100, 0.10, 0)

        np.testing.assert_array_equal(plan.rso_ids, expected_ids)
        for event in plan.events:
            self.assertEqual(event.time_seconds, 900.0)
            self.assertEqual(event.delta_v_rtn_mps, (0.0, 10.0, 0.0))

    def test_zero_fraction_gives_an_empty_plan(self):
        plan = ManeuverConfiguration(maneuvering_fraction=0.0).build_plan(100, 0)
        self.assertEqual(plan.events, ())
        self.assertEqual(len(plan.rso_ids), 0)

    def test_burn_time_outside_the_episode_is_rejected(self):
        with self.assertRaises(ValueError):
            ManeuverConfiguration(maneuvering_fraction=0.1, burn_time_seconds=EPISODE_LENGTH_SECONDS)


class RandomConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.configuration = RandomManeuverConfiguration(maneuvering_fraction=0.20)

    def test_tagged_cohort_matches_the_fixed_plan_at_the_same_fraction_and_seed(self):
        """Comparability with the existing fixed-delta-v runs depends on this."""
        random_plan = self.configuration.build_plan(100, 5)
        fixed_plan = ManeuverConfiguration(maneuvering_fraction=0.20).build_plan(100, 5)
        np.testing.assert_array_equal(random_plan.rso_ids, fixed_plan.rso_ids)

    def test_plan_is_reproducible_and_seed_dependent(self):
        first = self.configuration.build_plan(100, 5)
        repeat = self.configuration.build_plan(100, 5)
        other = self.configuration.build_plan(100, 6)
        self.assertEqual(first.events, repeat.events)
        self.assertNotEqual(first.events, other.events)

    def test_events_respect_the_declared_magnitude_and_time_bounds(self):
        for seed in range(10):
            for event in self.configuration.build_plan(100, seed).events:
                self.assertGreaterEqual(event.magnitude_mps, 1.0 - 1e-9)
                self.assertLessEqual(event.magnitude_mps, 100.0 + 1e-9)
                self.assertGreaterEqual(event.time_seconds, 600.0)
                self.assertLessEqual(event.time_seconds, 4800.0)

    def test_directions_are_isotropic_rather_than_along_track(self):
        directions = np.array(
            [
                np.asarray(event.delta_v_rtn_mps) / event.magnitude_mps
                for seed in range(60)
                for event in self.configuration.build_plan(100, seed).events
            ]
        )
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-12)
        # An isotropic draw has zero mean on every axis; the along-track-only plan would
        # instead put the whole mass on the transverse component.
        np.testing.assert_allclose(directions.mean(axis=0), np.zeros(3), atol=0.06)
        np.testing.assert_allclose(
            np.mean(directions**2, axis=0), np.full(3, 1.0 / 3.0), atol=0.05
        )

    def test_magnitudes_are_log_uniform(self):
        magnitudes = np.array(
            [
                event.magnitude_mps
                for seed in range(60)
                for event in self.configuration.build_plan(100, seed).events
            ]
        )
        log_magnitudes = np.log10(magnitudes)
        # log10 of a log-uniform draw over [1, 100] is uniform over [0, 2].
        self.assertAlmostEqual(float(log_magnitudes.mean()), 1.0, delta=0.05)
        self.assertAlmostEqual(
            float(log_magnitudes.std()), 2.0 / math.sqrt(12.0), delta=0.05
        )

    def test_null_burn_control_matches_the_treatment_except_for_magnitude(self):
        """The control must share cohort, burn times and directions, or the pairing is void.

        Building the control from a fixed burn time instead would hand the tagged objects a
        different post-burn observation window, which by itself changes reacquisition.
        """
        control = RandomManeuverConfiguration(maneuvering_fraction=0.20, magnitude_scale=0.0)
        for seed in range(8):
            treatment_plan = self.configuration.build_plan(100, seed)
            control_plan = control.build_plan(100, seed)

            np.testing.assert_array_equal(control_plan.rso_ids, treatment_plan.rso_ids)
            for treated, controlled in zip(treatment_plan.events, control_plan.events):
                self.assertEqual(controlled.rso_id, treated.rso_id)
                self.assertEqual(controlled.time_seconds, treated.time_seconds)
                self.assertEqual(controlled.magnitude_mps, 0.0)

    def test_null_burn_control_keeps_the_treatment_burn_direction(self):
        half = RandomManeuverConfiguration(maneuvering_fraction=0.20, magnitude_scale=0.5)
        treatment_plan = self.configuration.build_plan(100, 3)
        halved_plan = half.build_plan(100, 3)
        for treated, halved in zip(treatment_plan.events, halved_plan.events):
            np.testing.assert_allclose(
                np.asarray(halved.delta_v_rtn_mps),
                np.asarray(treated.delta_v_rtn_mps) * 0.5,
                rtol=1e-12,
            )

    def test_negative_magnitude_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            RandomManeuverConfiguration(maneuvering_fraction=0.1, magnitude_scale=-1.0)

    def test_invalid_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            RandomManeuverConfiguration(
                maneuvering_fraction=0.1, magnitude_range_mps=(0.0, 10.0)
            )
        with self.assertRaises(ValueError):
            RandomManeuverConfiguration(
                maneuvering_fraction=0.1, burn_time_window_seconds=(4800.0, 600.0)
            )


class ExplicitConfigurationTests(unittest.TestCase):
    def test_plan_is_returned_unchanged(self):
        plan = plan_from_manifest_events(
            [{"rso_id": 1, "time_seconds": 700.0, "delta_v_mps": [0.0, 3.0, 4.0]}]
        )
        configuration = ExplicitManeuverConfiguration(plan=plan)
        self.assertEqual(configuration.build_plan(100, 42), plan)
        self.assertAlmostEqual(configuration.maneuvering_fraction, 0.01)
        self.assertEqual(configuration.summary()["plan_type"], "explicit")


class FiniteBurnPlanTests(unittest.TestCase):
    def setUp(self):
        self.impulsive = RandomManeuverConfiguration(maneuvering_fraction=0.20)
        self.finite = RandomManeuverConfiguration(
            maneuvering_fraction=0.20, burn_duration_seconds=1800.0
        )

    def test_finite_plan_matches_the_impulsive_plan_except_for_duration(self):
        """Only the burn model may differ, or the impulsive-versus-finite contrast is confounded."""
        for seed in range(6):
            impulsive_plan = self.impulsive.build_plan(100, seed)
            finite_plan = self.finite.build_plan(100, seed)
            for impulse, finite in zip(impulsive_plan.events, finite_plan.events):
                self.assertEqual(finite.rso_id, impulse.rso_id)
                self.assertEqual(finite.time_seconds, impulse.time_seconds)
                self.assertEqual(finite.delta_v_rtn_mps, impulse.delta_v_rtn_mps)
                self.assertEqual(impulse.burn_duration_seconds, 0.0)
                self.assertEqual(finite.burn_duration_seconds, 1800.0)

    def test_impulsive_events_keep_the_original_manifest_shape(self):
        event = self.impulsive.build_plan(100, 0).events[0]
        self.assertEqual(set(event.to_dict()), {"rso_id", "time_seconds", "delta_v_mps"})

    def test_finite_events_declare_their_duration_in_the_manifest(self):
        event = self.finite.build_plan(100, 0).events[0]
        self.assertEqual(event.to_dict()["burn_duration_seconds"], 1800.0)

    def test_manifest_round_trip_preserves_duration(self):
        plan = self.finite.build_plan(100, 0)
        self.assertEqual(plan_from_manifest_events(plan.to_dicts()).events, plan.events)

    def test_negative_duration_is_rejected(self):
        with self.assertRaises(ValueError):
            RandomManeuverConfiguration(
                maneuvering_fraction=0.1, burn_duration_seconds=-1.0
            )

    def test_summary_names_the_burn_model(self):
        self.assertEqual(self.impulsive.summary()["burn_model"], "impulsive")
        self.assertEqual(self.finite.summary()["burn_model"], "finite")

if __name__ == "__main__":
    unittest.main()
