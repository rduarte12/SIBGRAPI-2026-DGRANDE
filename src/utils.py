import numpy as np
from sklearn.model_selection import StratifiedKFold


def compute_mean_std(values):
    """Compute the mean and sample standard deviation of a list of values.

    Args:
        values: Sequence of numeric values.

    Returns:
        Tuple ``(mean, std)`` where ``std`` uses ``ddof=1`` (sample standard
        deviation).
    """
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    return mean, std


def create_folds(features, labels, n_folds=10):
    """Build stratified cross-validation folds.

    Args:
        features: Feature matrix ``(n_samples, n_features)``.
        labels: Label vector ``(n_samples,)``.
        n_folds: Number of stratified splits.

    Returns:
        List of ``(test_index, train_index)`` tuples, following the original
        ``StratifiedKFold.split`` ordering (note the split returns
        ``(train, test)``, so the first element of each tuple here is used as
        the test set downstream).
    """
    kf = StratifiedKFold(n_splits=n_folds, shuffle=False)
    return list(kf.split(features, labels))
