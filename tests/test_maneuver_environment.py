from pathlib import Path
import sys
import unittest

import numpy as np

from real_environment.maneuver_environment import (
    ManeuverConfiguration,
    filter_maneuver_measurement_indices,
    select_maneuvering_rso_ids,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SMITH_SOURCE = REPOSITORY_ROOT / "try_code" / "MARLSSA-main"
if str(SMITH_SOURCE) not in sys.path:
    sys.path.insert(0, str(SMITH_SOURCE))


class ManeuverEnvironmentTests(unittest.TestCase):
    def test_selection_is_reproducible_and_without_replacement(self):
        first = select_maneuvering_rso_ids(100, 0.10, 7)
        second = select_maneuvering_rso_ids(100, 0.10, 7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(len(np.unique(first)), 10)

    def test_different_base_seed_changes_selected_subset(self):
        first = select_maneuvering_rso_ids(100, 0.10, 7)
        second = select_maneuvering_rso_ids(100, 0.10, 8)
        self.assertFalse(np.array_equal(first, second))

    def test_truth_gate_only_filters_maneuvering_objects(self):
        filtered = filter_maneuver_measurement_indices(
            stock_indices=[1, 2, 3, 4],
            maneuvering_ids={2, 3},
            truth_fov_mask_by_rso={2: False, 3: True},
        )
        np.testing.assert_array_equal(filtered, np.array([1, 3, 4]))

    def test_configuration_rejects_invalid_fraction(self):
        with self.assertRaises(ValueError):
            ManeuverConfiguration(maneuvering_fraction=1.1)


if __name__ == "__main__":
    unittest.main()
