from typing import Optional

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
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


def compute_uncertainty(
    adata: anndata.AnnData, obs_key: str, use_rep: Optional[str] = None, n_neighbors: Optional[int] = 10
):
    """Evaluate uncertainty of prediction using the given obs.

    Parameters
    ----------
    adata
        Annotated data object with `adata.X` model latent representations.
    obs_key
        Key in :attr:`anndata.AnnData.obs` used to evaluate the uncertainty with respect to.
    use_rep
        Representation to use for neighbors calculation.
    n_neighbors
        Number of neighbors to consider.

    Returns
    -------
    Adds "uncertainty" key to :attr:`anndata.AnnData.obs`.

    Notes
    -----
    Based on chemCPA uncertainty definition.
    """
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)

    distances = adata.obsp["distances"].A
    neighbours = np.where(distances != 0)

    def entropy(obs, base=None):
        vc = pd.Series(obs).value_counts(normalize=True, sort=False)
        base = np.exp if base is None else base
        return -(vc * np.log(vc) / np.log(base)).sum()

    for i in range(adata.shape[0]):
        cond = neighbours[0] == i
        adata.obs[obs_key].iloc[i]
        key_neigh = adata.obs[obs_key].iloc[neighbours[1][cond]]
        adata.obs.loc[adata.obs_names == adata.obs_names[i], "uncertainty"] = (
            1 / np.log(distances[i].sum()) * entropy(key_neigh, base=2)
        )
    return adata
