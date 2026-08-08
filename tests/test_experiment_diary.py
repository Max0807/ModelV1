from __future__ import annotations

import unittest

from modelv1.experiment_diary import _describe_loss


class ExperimentDiaryTests(unittest.TestCase):
    def test_describes_legacy_direct_uv_loss(self) -> None:
        description = _describe_loss(
            {"beta_mm": 30.0, "gate_regularization_weight": 0.001}
        )

        self.assertEqual(description, "beta_mm=30.0，gate_reg=0.001")

    def test_describes_v4_gaze_geometry_loss_without_legacy_beta(self) -> None:
        description = _describe_loss(
            {
                "uv_huber_beta_mm": 30.0,
                "uv_huber_weight": 1.0,
                "gaze_angular_weight": 100.0,
                "ray_validity_weight": 0.1,
            }
        )

        self.assertEqual(
            description,
            "uv_huber_beta_mm=30.0，uv_huber_weight=1.0，"
            "gaze_angular_weight=100.0，ray_validity_weight=0.1",
        )


if __name__ == "__main__":
    unittest.main()
