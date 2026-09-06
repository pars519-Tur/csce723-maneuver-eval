import unittest

import numpy as np

from real_environment.policy_adapters import AdvancedGreedyAdapter


class PolicyAdapterTests(unittest.TestCase):
    def test_advanced_greedy_selects_unique_highest_score_cell(self):
        observation = np.zeros((271, 19, 11), dtype=float)
        observation[90 + 44, 6, 10] = 1.0
        observation[90 + 50, 8, 9] = 10.0
        action = AdvancedGreedyAdapter().actions({"agent_0": observation})["agent_0"]
        self.assertEqual(action, 50 * 19 + 8)


if __name__ == "__main__":
    unittest.main()
