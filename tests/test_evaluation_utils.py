import unittest

import numpy as np
import torch

from cse_smnet.utils.memory import scheduled_memory_temperature
from cse_smnet.utils.metrics import (
    build_frame_labels,
    compute_auc_eer,
    compute_psnr,
    normalize_scores,
    psnr_to_anomaly_score,
)


class EvaluationUtilityTests(unittest.TestCase):
    def test_frame_labels_are_inclusive(self):
        labels = build_frame_labels(6, [(2, 4)], one_based=True)
        np.testing.assert_array_equal(labels, np.array([0, 1, 1, 1, 0, 0]))

    def test_psnr_to_anomaly_score(self):
        scores = psnr_to_anomaly_score(np.array([10.0, 20.0]))
        np.testing.assert_allclose(scores, np.array([-10.0, -20.0]))

    def test_normalization(self):
        normalized = normalize_scores(np.array([3.0, 1.0, 2.0]))
        np.testing.assert_allclose(normalized, np.array([1.0, 0.0, 0.5]))

    def test_psnr(self):
        pred = torch.tensor([[[[0.0, 1.0], [0.5, 0.25]]]])
        target = torch.tensor([[[[0.0, 0.5], [0.5, 0.25]]]])
        mse = ((pred - target) ** 2).mean((1, 2, 3))
        expected = 10.0 * torch.log10(1.0 / (mse + 1e-8))
        self.assertTrue(torch.allclose(compute_psnr(pred, target), expected))

    def test_auc_eer(self):
        auc, eer, threshold = compute_auc_eer(
            np.array([0.1, 0.2, 0.8, 0.9]),
            np.array([0, 0, 1, 1]),
        )
        self.assertAlmostEqual(auc, 1.0)
        self.assertAlmostEqual(eer, 0.0)
        self.assertTrue(np.isfinite(threshold))

    def test_temperature_schedule_endpoints(self):
        self.assertAlmostEqual(
            scheduled_memory_temperature(0, 100, 0.30, 0.10, "cosine"),
            0.30,
        )
        self.assertAlmostEqual(
            scheduled_memory_temperature(99, 100, 0.30, 0.10, "cosine"),
            0.10,
        )


if __name__ == "__main__":
    unittest.main()
