from typing import Optional

import numpy as np
import torch


def repeat_n(x, n, device=None):
    """Returns an n-times repeated version of the Tensor x, repetition dimension is axis 0."""
    device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    return x.to(device).view(1, -1).repeat(n, 1)


def biolord_metric(
    r2_mean: float,
    r2_var: float,
    classification_accuracy: Optional[float] = np.nan,
) -> float:
    """Evaluate biolord metric.

    Parameters
    ----------
    r2_mean
        r2 score of the mean of the gene expression
    r2_var
        r2 score of the variance of the gene expression
    classification_accuracy
        classification accuracy of classify module

    Returns
    -------
    mean of input values.
    """
    return np.nanmean([r2_mean, r2_var, classification_accuracy])
