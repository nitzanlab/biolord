from typing import Any, Literal

import torch
from torch import nn

__all__ = ["DistributionDecoderWrapper", "NormalDecoderWrapper", "RegularizedEmbedding"]


# Decoder
class DistributionDecoderWrapper(nn.Module):
    """Decodes data from latent space of ``n_hidden`` dimensions into ``n_output`` dimensions.

    Parameters
    ----------
    n_output
        The dimensionality of the output (data space)
    n_hidden
        The number of nodes per hidden layer
    scale_activation
        Activation layer to use for px_scale_decoder
    """

    def __init__(
        self,
        n_output: int,
        n_hidden: int = 128,
        scale_activation: Literal["softmax", "softplus"] = "softmax",
        **kwargs: Any,
    ):
        super().__init__()

        # mean gamma
        if scale_activation == "softmax":
            px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            px_scale_activation = nn.Softplus()
        else:
            raise ValueError(f"{scale_activation} is not implemented as a valid activation.")

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            px_scale_activation,
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        dispersion
            One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        z
            tensor with shape ``(n_hidden,)``
        library
            size of library

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression

        """
        # The decoder returns values for the parameters of the ZINB distribution
        px_scale = self.px_scale_decoder(z)
        px_dropout = self.px_dropout_decoder(z)
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r_decoder(z) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


class NormalDecoderWrapper(nn.Module):
    """Decodes data from hidden space to data space.

    ``n_hidden`` dimensions to ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    Output pear feature is according to defined distribution:
    "normal": mean, variance

    Parameters
    ----------
    n_output
        The dimensionality of the output (data space)
    n_hidden
        The number of nodes per hidden layer
    """

    def __init__(self, n_output: int, n_hidden: int = 128, **kwargs: Any):
        super().__init__()

        self.mean_decoder = nn.Linear(n_hidden, n_output)
        self.var_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        Parameters
        ----------
        x
            tensor with shape ``(n_hidden,)``

        Returns
        -------
        2-tuple of :py:class:`torch.Tensor`
            Mean and variance tensors of shape ``(n_output,)``

        """
        # Parameters for latent distribution
        p_m = self.mean_decoder(x)
        p_v = torch.exp(self.var_decoder(x))
        return p_m, p_v


class RegularizedEmbedding(nn.Module):
    """Regularized embedding module."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        sigma: float,
        embed: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=n_input,
            embedding_dim=n_output,
        )
        self.sigma = sigma if embed else 0
        self.embed = embed

    def forward(self, x):
        """Forward pass."""
        x_ = self.embedding(x)
        if self.training and self.sigma != 0:
            noise = torch.zeros_like(x_)
            noise.normal_(mean=0, std=self.sigma)

            x_ = x_ + noise
        x_ = x_ * self.embed
        return x_
