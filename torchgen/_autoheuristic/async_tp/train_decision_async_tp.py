# mypy: ignore-errors
"""
Training script for async TP fuse/no_fuse decision tree heuristic.

Inherits from AHTrainDecisionTree to leverage the standard AutoHeuristic
training pipeline: grid search, unsafe leaf detection, confidence thresholds,
and artifact code generation.

Usage:
    cd torchgen/_autoheuristic/async_tp
    python train_decision_async_tp.py async_tp_h100_data.txt --heuristic-name AsyncTPFuseH100
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).absolute().parents[1]))

from train_decision import AHTrainDecisionTree


class AHTrainDecisionTreeAsyncTP(AHTrainDecisionTree):
    def __init__(self):
        super().__init__()

    def add_new_features(self, results):
        """
        Features m, k, n, arith_intensity, m_times_k, m_times_n, k_times_n
        are already computed in the data by convert_benchmark_data.py.
        No additional feature engineering needed.
        """
        return (results, [])

    def get_default_config(self, row):
        """Default config when AutoHeuristic returns 'unsure'."""
        return "no_fuse"

    def get_allowed_wrong_prediction_pct(self):
        """
        For async TP, wrong predictions can cause 2-3x regressions in decode phase.
        Use a stricter threshold than mixed_mm.
        """
        return 0.02

    def get_test_and_val_size(self):
        """
        Our dataset is relatively small (~116 unique configs), so use a larger
        validation set proportion for better model selection.
        """
        return (0.15, 0.20)

    def is_unsafe_leaf(self, row, predicted_config, choice2time):
        """
        Mark a leaf as unsafe if the predicted choice is significantly slower
        than the best choice. For async TP, fusing when we shouldn't can cause
        2-3x regression in decode, so we are conservative.

        If the leaf is unsafe, the heuristic returns None (unsure), and the
        caller falls back to no_fuse (safe default).
        """
        if predicted_config not in choice2time:
            return False

        predicted_time = choice2time[predicted_config]
        best_time = min(choice2time.values())

        # If predicted choice is >10% slower than best, mark unsafe
        if predicted_time > 1.10 * best_time:
            return True

        # Also unsafe if we predict fuse but no_fuse is better by >5%
        if predicted_config == "fuse" and "no_fuse" in choice2time:
            no_fuse_time = choice2time["no_fuse"]
            if predicted_time > 1.05 * no_fuse_time:
                return True

        return False

    def get_grid_search_values(self):
        """
        Grid search over hyperparameters. Use a wider depth range since our
        decision boundary involves multiple features (m, k, n, m_times_n).
        """
        return {
            "max_depth": [3, 4, 5, 6, 7],
            "min_samples_leaf": [1, 2, 5, 0.05],
            "criterion": ["gini", "entropy"],
        }


if __name__ == "__main__":
    train = AHTrainDecisionTreeAsyncTP()
    train.generate_heuristic()
