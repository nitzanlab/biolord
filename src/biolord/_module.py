from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from scvi import REGISTRY_KEYS, settings
from scvi.distributions import NegativeBinomial, Poisson
from scvi.module import Classifier
from scvi.module.base import BaseModuleClass, auto_move_data
from scvi.nn import Decoder, DecoderSCVI, FCLayers
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn
from torch.distributions import Categorical, Normal

from ._constants import LOSS_KEYS

__all__ = ["RegularizedEmbedding", "BiolordModule", "BiolordClassifyModule"]


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


class BiolordModule(BaseModuleClass):
    """The :mod:`biolord` module.

    Parameters
    ----------
    n_genes
        Number of input genes.
    n_samples
        Number of layers.
    x_loc
        The expression data location.
    ordered_attributes_map
        Dictionary of ordered classes and their dimensions.
    categorical_attributes_map
        Dictionary for categorical classes, containing categorical values with keys as each category name and values
        as the categorical integer assignment.
    n_latent
        Latent dimension.
    n_latent_attribute_ordered
        Latent dimension of ordered attributes.
    n_latent_attribute_categorical
        Latent dimension of categorical attributes.
    gene_likelihood
        The gene_likelihood model.
    reconstruction_penalty
        MSE error to reconstruction loss.
    use_batch_norm
        Use batch norm in layers.
    use_layer_norm
        Use layer norm in layers.
    unknown_attribute_noise_param
        Noise strength added to encoding of unknown attributes.
    unknown_attributes
        Whether to include learning for unknown attributes
    attribute_dropout_rate
        Dropout rate.
    attribute_nn_width
        Ordered attributes autoencoder layers' width.
    attribute_nn_depth
        Ordered attributes autoencoder number of layers.
    attribute_nn_activation
        Use activation in ordered attributes.
    decoder_width
        Decoder layers' width.
    decoder_depth
        Decoder number of layers.
    decoder_activation
        Use activation in decoder.
    eval_r2_ordered
        Evaluate the R2 w.r.t. the ordered attribute. Set to `True` only if ordered attributes are binned.
    decoder_dropout_rate
        Decoder dropout rate.
    seed
        Random seed.
    """

    def __init__(
        self,
        n_genes: int,
        n_samples: int,
        x_loc: str,
        ordered_attributes_map: Optional[Dict[str, int]] = None,
        categorical_attributes_map: Optional[Dict[str, Dict]] = None,
        n_latent: int = 32,
        n_latent_attribute_categorical: int = 4,
        n_latent_attribute_ordered: int = 16,
        gene_likelihood: Literal["normal", "nb", "poisson"] = "normal",
        reconstruction_penalty: float = 1e2,
        unknown_attribute_penalty: float = 1e1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        unknown_attribute_noise_param: float = 1e-1,
        unknown_attributes: bool = True,
        attribute_dropout_rate: Dict[str, float] = None,
        decoder_width: int = 512,
        decoder_depth: int = 4,
        decoder_activation: bool = True,
        attribute_nn_width: Dict[str, int] = None,
        attribute_nn_depth: Dict[str, int] = None,
        attribute_nn_activation: bool = True,
        eval_r2_ordered: bool = False,
        decoder_dropout_rate: float = 0.1,
        seed: int = 0,
    ):
        super().__init__()
        gene_likelihood = gene_likelihood.lower()
        assert gene_likelihood in ["normal", "nb", "poisson"], gene_likelihood

        default_width = 256
        default_depth = 2
        torch.manual_seed(seed)
        np.random.seed(seed)
        settings.seed = seed

        self.ae_loss_fn = nn.GaussianNLLLoss()
        self.ae_loss_mse_fn = nn.MSELoss()
        self.reconstruction_penalty = reconstruction_penalty
        self.unknown_attribute_penalty = unknown_attribute_penalty
        self.mm_regression_loss_fn = nn.BCEWithLogitsLoss()

        self.n_genes = n_genes
        self.n_latent = n_latent
        self.x_loc = x_loc
        self.n_latent_attribute_categorical = n_latent_attribute_categorical
        self.n_latent_attribute_ordered = n_latent_attribute_ordered
        self.gene_likelihood = gene_likelihood
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.eval_r2_ordered = eval_r2_ordered

        self.n_decoder_input = n_latent + (
            n_latent_attribute_categorical * len(categorical_attributes_map)
            + n_latent_attribute_ordered * len(ordered_attributes_map)
        )
        self.categorical_attributes_map = (
            categorical_attributes_map if isinstance(categorical_attributes_map, Dict) else {}
        )
        self.ordered_attributes_map = ordered_attributes_map if isinstance(ordered_attributes_map, Dict) else {}

        if isinstance(attribute_nn_width, Dict):
            self.attribute_nn_width = attribute_nn_width
        elif attribute_nn_width is None:
            self.attribute_nn_width = {attribute_: default_width for attribute_ in self.ordered_attributes_map}
        else:
            self.attribute_nn_width = {attribute_: attribute_nn_width for attribute_ in self.ordered_attributes_map}

        if isinstance(attribute_nn_depth, Dict):
            self.attribute_nn_depth = attribute_nn_depth
        elif attribute_nn_depth is None:
            self.attribute_nn_depth = {attribute_: default_depth for attribute_ in self.ordered_attributes_map}
        else:
            self.attribute_nn_depth = {attribute_: attribute_nn_depth for attribute_ in self.ordered_attributes_map}

        if isinstance(attribute_dropout_rate, Dict):
            self.attribute_dropout_rate = attribute_dropout_rate
        elif attribute_dropout_rate is None:
            self.attribute_dropout_rate = {
                attribute_: decoder_dropout_rate for attribute_ in self.ordered_attributes_map
            }
        else:
            self.attribute_dropout_rate = {
                attribute_: attribute_dropout_rate for attribute_ in self.ordered_attributes_map
            }

        self.latent_codes = RegularizedEmbedding(
            n_input=n_samples, n_output=n_latent, sigma=unknown_attribute_noise_param, embed=unknown_attributes
        )

        # Create Embeddings
        # 1. ordered classes
        reps_ordered = []
        self.ordered_networks = nn.ModuleDict()
        for attribute_, len_ in self.ordered_attributes_map.items():
            if "_rep" in attribute_:
                reps_ordered.append(attribute_)
            else:
                self.ordered_networks[attribute_] = FCLayers(
                    n_in=len_,
                    n_out=self.n_latent_attribute_ordered,
                    n_layers=self.attribute_nn_depth[attribute_],
                    n_hidden=self.attribute_nn_width[attribute_],
                    dropout_rate=self.attribute_dropout_rate[attribute_],
                    bias=False,
                    use_activation=attribute_nn_activation,
                )
        for attribute_ in reps_ordered:
            self.ordered_networks[attribute_] = self.ordered_networks[attribute_.split("_rep")[0]]

        # 2. categorical classes
        self.categorical_embeddings = nn.ModuleDict()
        reps_categorical = []
        for attribute_, unique_categories in self.categorical_attributes_map.items():
            if "_rep" in attribute_:
                reps_categorical.append(attribute_)
            else:
                self.categorical_embeddings[attribute_] = torch.nn.Embedding(
                    len(unique_categories),
                    n_latent_attribute_categorical,
                )
        for attribute_ in reps_categorical:
            self.categorical_embeddings[attribute_] = self.categorical_embeddings[attribute_.split("_rep")[0]]

        # Decoder components
        if self.gene_likelihood in ["nb", "poisson"]:
            self.decoder = DecoderSCVI(
                n_input=self.n_decoder_input,
                n_output=n_genes,
                n_hidden=decoder_width,
                n_layers=decoder_depth,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                scale_activation="softmax",
            )
            self.px_r = torch.nn.Parameter(torch.randn(n_genes))
        else:
            self.decoder = Decoder(
                n_input=self.n_decoder_input,
                n_output=n_genes,
                n_hidden=decoder_width,
                n_layers=decoder_depth,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                use_activation=decoder_activation,
            )

    def _get_inference_input(self, tensors: Dict[Any, Any], **kwargs):
        x = tensors[self.x_loc]  # batch_size, n_genes
        sample_indices = tensors[REGISTRY_KEYS.INDICES_KEY].long().ravel()

        categorical_attribute_dict = {}
        for attribute_ in self.categorical_attributes_map:
            categorical_attribute_dict[attribute_] = tensors[attribute_].view(
                -1,
            )

        ordered_attribute_dict = {}
        for attribute_ in self.ordered_attributes_map:
            ordered_attribute_dict[attribute_] = tensors[attribute_]

        input_dict = {
            "genes": x,
            "sample_indices": sample_indices,
            "categorical_attribute_dict": categorical_attribute_dict,
            "ordered_attribute_dict": ordered_attribute_dict,
        }
        return input_dict

    def get_inference_input(self, tensors: Dict[Any, Any], **kwargs) -> Dict[str, Any]:
        """Convert tensors to valid inference input.

        Parameters
        ----------
        tensors
            Considered inputs.
        kwargs
            Additional arguments

        Returns
        -------
        Dictionary with the module's expected input tensors (`genes`, `sample_indices`, `categorical_attribute_dict`, and `ordered_attribute_dict`).
        """
        return self._get_inference_input(tensors, **kwargs)

    @auto_move_data
    def _inference_attribute_embeddings(
        self,
        genes,
        categorical_attribute_dict,
        ordered_attribute_dict,
        nullify_attribute=None,
    ):
        """Inference over attribute embeddings."""
        nullify_attribute = [] if nullify_attribute is None else nullify_attribute
        inference_output = {}
        batch_size = genes.shape[0]
        for attribute_, embedding_ in self.categorical_embeddings.items():
            latent_i = embedding_(categorical_attribute_dict[attribute_].long())
            latent_i = latent_i.view(batch_size, self.n_latent_attribute_categorical).unsqueeze(
                0
            )  # 1, batch_size, n_latent_attribute_categorical
            if attribute_ in nullify_attribute:
                latent_i = torch.zeros_like(latent_i)
            inference_output[attribute_] = latent_i

        for attribute_, network_ in self.ordered_networks.items():
            latent_i = network_(ordered_attribute_dict[attribute_])
            latent_i = latent_i.view(batch_size, self.n_latent_attribute_ordered).unsqueeze(0)
            if attribute_ in nullify_attribute:
                latent_i = torch.zeros_like(latent_i)
            inference_output[attribute_] = latent_i

        return inference_output

    def _get_latent_unknown_attributes(
        self,
        sample_indices,
    ):
        """Get the module's latent unknown attributes representation."""
        latent_unknown_attributes = self.latent_codes(sample_indices)

        return latent_unknown_attributes

    @auto_move_data
    def inference(
        self,
        genes: torch.Tensor,
        sample_indices: torch.Tensor,
        categorical_attribute_dict: Dict[Any, Any],
        ordered_attribute_dict: Dict[Any, Any],
        nullify_attribute: Optional[List] = None,
    ) -> Dict[str, Any]:
        """Apply module inference.

        Parameters
        ----------
        genes
            Input expression.
        sample_indices
            Indices in the :class:`~anndata.AnnData` object of the input samples.
        categorical_attribute_dict
            Dictionary with categorical attributes as keys and the attribute sample labels as values.
        ordered_attribute_dict
            Dictionary with ordered attributes as keys and the attribute sample values as values.
        nullify_attribute
            Attributes to exclude from inferred latent space.

        Returns
        -------
        Dictionary with the module's expected input tensors (`genes`, `sample_indices`, `categorical_attribute_dict`, and `ordered_attribute_dict`).
        """
        nullify_attribute = [] if nullify_attribute is None else nullify_attribute
        inference_output = {}
        x_ = genes
        library = torch.log(genes.sum(1)).unsqueeze(1)

        latent_unknown_attributes = self._get_latent_unknown_attributes(sample_indices=sample_indices)

        latent_classes = self._inference_attribute_embeddings(
            genes=x_,
            categorical_attribute_dict=categorical_attribute_dict,
            ordered_attribute_dict=ordered_attribute_dict,
            nullify_attribute=nullify_attribute,
        )

        latent_vecs = [latent_unknown_attributes.squeeze()]
        for key_, latent_ in latent_classes.items():
            latent_vecs.append(latent_.squeeze())
            inference_output[key_] = latent_.squeeze()

        latent = torch.cat(latent_vecs, dim=-1)

        inference_output["latent"] = latent
        inference_output["latent_unknown_attributes"] = latent_unknown_attributes
        inference_output["library"] = library

        return inference_output

    def _get_generative_input(self, tensors, inference_outputs, **kwargs):
        input_dict = {
            "latent": inference_outputs["latent"],
            "library": inference_outputs["library"],
        }
        return input_dict

    @auto_move_data
    def generative(
        self,
        latent: torch.Tensor,
        library: torch.Tensor = None,
    ) -> Dict[str, Any]:
        """Runs the generative step.

        Parameters
        ----------
        latent
            The concatenated decomposed latent space.
        library
            Library sizes for each cell.

        Returns
        -------
        Dictionary with the generative predictions of the expression distribution.
        """
        if self.gene_likelihood in ["nb", "poisson"]:
            px_scale, _, px_rate, _ = self.decoder(
                dispersion="gene",
                z=latent,
                library=library,
            )
            px_r = torch.exp(self.px_r)
            px = (
                NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
                if self.gene_likelihood == "nb"
                else Poisson(px_rate)
            )  # , scale=px_scale)

            return {
                "means": px.mean,
                "variances": px.variance,
                "distribution": px,
                "samples": px.sample().squeeze(0),
            }

        else:
            p_m, p_v = self.decoder(x=latent)
            px = Normal(loc=p_m, scale=p_v.sqrt())
            return {
                "means": px.loc,
                "variances": px.variance,
                "distribution": px,
                "samples": px.sample().squeeze(0),
            }

    @auto_move_data
    def loss(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[Literal["latent_unknown_attributes"], torch.Tensor],
        generative_outputs: Dict[Literal["distribution", "means", "variances"], torch.Tensor],
    ) -> Dict[str, float]:
        """Computes the module's loss.

        Parameters
        ----------
        tensors
            Considered model inputs.
        inference_outputs
            Inference step outputs.
        generative_outputs
            Generative step outputs.

        Returns
        -------
        The loss elements.
        """
        x_ = tensors[self.x_loc]
        means = generative_outputs["means"]
        variances = generative_outputs["variances"]

        if self.gene_likelihood in ["nb", "poisson"]:
            reconstruction_loss = -generative_outputs["distribution"].log_prob(x_).sum(-1)
            reconstruction_loss = reconstruction_loss.mean()
        else:
            reconstruction_loss = self.ae_loss_fn(input=means, target=x_, var=variances)

        reconstruction_loss += self.reconstruction_penalty * self.ae_loss_mse_fn(input=means, target=x_)

        unknown_attribute_penalty_loss_val = self.unknown_attribute_penalty_loss(
            inference_outputs["latent_unknown_attributes"]
        )

        return {
            LOSS_KEYS.RECONSTRUCTION: reconstruction_loss,
            LOSS_KEYS.UNKNOWN_ATTRIBUTE_PENALTY: unknown_attribute_penalty_loss_val,
        }

    @staticmethod
    def unknown_attribute_penalty_loss(latent_unknown_attributes: torch.Tensor) -> float:
        """Computes the content penalty term in the loss."""
        return torch.sum(latent_unknown_attributes**2, dim=1).mean()

    @torch.no_grad()
    def r2_metric(
        self,
        tensors: Dict[str, torch.Tensor],
        generative_outputs: Dict[str, torch.Tensor],
    ) -> Tuple[float, float]:
        """Evaluate the :math:`R^2` metric over gene expression.

        Parameters
        ----------
        tensors
            Considered inputs.
        generative_outputs
            Generative model outputs.

        Returns
        -------
        The :math:`R^2` of the mean and standard deviation predictions of the gene expression.
        """
        x = tensors[self.x_loc].detach().cpu().numpy()  # batch_size, n_genes

        batch_size = x.shape[0]
        indices = torch.zeros(batch_size).to(self.device)
        if self.eval_r2_ordered:
            for ordered_attribute_, len_ in self.ordered_attributes_map.items():
                if len_ > 1:
                    attribute_vals = tensors[ordered_attribute_]  # (batch_size, n_class)
                    indices += (
                        attribute_vals
                        * torch.arange(attribute_vals.shape[1])
                        .view(1, -1)
                        .repeat(batch_size, 1)
                        .to(attribute_vals.device)
                    ).sum(dim=1)
                else:
                    indices += tensors[ordered_attribute_].view(
                        -1,
                    )  # (batch_size,)

        for categorical_attribute_ in self.categorical_attributes_map:
            indices += tensors[categorical_attribute_].view(
                -1,
            )  # (batch_size,)

        unique_indices = indices.unique()

        r2_mean = 0.0
        r2_var = 0.0
        k = 0

        pred_x_mean = (
            torch.nan_to_num(generative_outputs["means"], nan=0, neginf=0, posinf=100).detach().cpu().numpy()
        )  # batch_size, n_genes
        pred_x_var = (
            torch.nan_to_num(generative_outputs["variances"], nan=0, neginf=0, posinf=100).detach().cpu().numpy()
        )  # batch_size, n_genes

        for index in unique_indices:
            index_mask = (indices == index).detach().cpu().numpy()
            if index_mask.sum() > 2:
                x_index = x[index_mask]
                means_index = pred_x_mean[index_mask]
                variances_index = pred_x_var[index_mask]

                true_mean_index = np.nanmean(x_index, axis=0)
                pred_mean_index = np.nanmean(means_index, axis=0)

                true_var_index = np.nanvar(x_index, axis=0)
                pred_var_index = (
                    np.nanvar(means_index, axis=0)
                    if self.gene_likelihood in ["nb", "poisson"]
                    else np.nanmean(variances_index, axis=0)
                )

                r2_mean += r2_score(true_mean_index, pred_mean_index)
                r2_var += r2_score(true_var_index, pred_var_index)
                k += 1
            else:
                continue
        if k > 0:
            return r2_mean / k, r2_var / k
        else:
            return r2_mean, r2_var

    @torch.no_grad()
    def get_expression(
        self, tensors: Dict[str, torch.Tensor], **inference_kwargs: Any
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Computes gene expression means and standard deviation.

        Parameters
        ----------
        tensors
            Considered inputs.
        inference_kwargs
            Additional arguments.

        Returns
        -------
        Prediction of gene expression mean and standard deviation.
        """
        _, generative_outputs = self.forward(
            tensors,
            compute_loss=False,
            inference_kwargs=inference_kwargs,
        )

        mus = torch.nan_to_num(generative_outputs["means"], nan=0, neginf=0, posinf=100)  # batch_size, n_genes
        stds = torch.nan_to_num(generative_outputs["variances"], nan=0, neginf=0, posinf=100)  # batch_size, n_genes
        return mus, stds


class BiolordClassifyModule(BiolordModule):
    """The `biolord-classify` module.

    A :class:`~biolord.BiolordModule` accompanied by regressors for ordered classes and
    classifiers for categorical classes.

    Parameters
    ----------
    categorical_attributes_missing
        Categorical categories representing un-labeled cells.
    classify_all
        Whether to classify all classes or only semi-supervised classes.
    logits
        Classifier output type.
    bias
        Whether to add bias to the regressor.
    classification_penalty
        Classification penalty strength.
    classifier_nn_width
        Classifier's layer width.
    classifier_nn_depth
        Classifier's number of layers.
    classifier_dropout_rate
        Classifier's dropout rate.
    loss_regression
        Loss function for regressors
    kwargs
        Keyword arguments for :class:`~biolord.BiolordModule`.
    """

    def __init__(
        self,
        categorical_attributes_missing: Optional[Dict[str, str]] = None,
        classify_all: bool = False,
        logits: bool = False,
        bias: bool = True,
        classification_penalty: float = 1e-1,
        classifier_penalty: float = 1e-4,
        classifier_nn_width: int = 128,
        classifier_nn_depth: int = 2,
        classifier_dropout_rate: float = 1e-1,
        loss_regression: Literal["normal", "mse"] = "normal",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        loss_regression = loss_regression.lower()
        assert loss_regression in ["normal", "mse"], loss_regression

        self.ae_loss_fn = nn.GaussianNLLLoss()
        self.ae_loss_mse_fn = nn.MSELoss()
        self.classification_penalty = classification_penalty
        self.classifier_penalty = classifier_penalty
        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.regression_loss_fn = nn.MSELoss() if loss_regression == "mse" else nn.GaussianNLLLoss()
        self.loss_regression = "mse"
        self.mm_regression_loss_fn = nn.BCEWithLogitsLoss()
        self.classify_all = classify_all

        if isinstance(categorical_attributes_missing, Dict):
            self.categorical_attributes_missing = categorical_attributes_missing
        elif categorical_attributes_missing is None:
            self.categorical_attributes_missing = {attribute_: None for attribute_ in self.categorical_attributes_map}
        else:
            self.categorical_attributes_missing = {
                attribute_: categorical_attributes_missing for attribute_ in self.categorical_attributes_map
            }

        self.categorical_embeddings = nn.ModuleDict()
        for attribute_, unique_categories in self.categorical_attributes_map.items():
            if self.categorical_attributes_missing[attribute_] is not None:
                self.categorical_embeddings[attribute_] = torch.nn.Embedding(
                    len(unique_categories),
                    self.n_latent_attribute_categorical,
                    padding_idx=self.categorical_attributes_map[attribute_][
                        self.categorical_attributes_missing[attribute_]
                    ],
                )
            else:
                self.categorical_embeddings[attribute_] = torch.nn.Embedding(
                    len(unique_categories),
                    self.n_latent_attribute_categorical,
                )

        # Create classifiers
        self.ordered_regressors = nn.ModuleDict()
        if self.classify_all:
            if self.loss_regression == "mse":
                self.ordered_regressors = nn.ModuleDict(
                    {
                        attribute_: nn.Linear(
                            in_features=self.n_genes,
                            out_features=len_,
                            bias=bias,
                        )
                        for attribute_, len_ in self.ordered_attributes_map.items()
                    }
                )
            else:
                self.ordered_regressors = nn.ModuleDict(
                    {
                        attribute_: Decoder(
                            n_input=self.n_genes,
                            n_output=len_,
                            n_hidden=classifier_nn_width,
                            n_layers=classifier_nn_depth,
                            use_batch_norm=self.use_batch_norm,
                            use_layer_norm=self.use_layer_norm,
                        )
                        for attribute_, len_ in self.ordered_attributes_map.items()
                    }
                )

        self.categorical_classifiers = nn.ModuleDict()
        for attribute_, unique_categories in self.categorical_attributes_map.items():
            if self.categorical_attributes_missing[attribute_] is not None:
                self.categorical_classifiers[attribute_] = Classifier(
                    n_input=self.n_genes,
                    n_labels=len(unique_categories) - 1,
                    n_hidden=classifier_nn_width,
                    n_layers=classifier_nn_depth,
                    dropout_rate=classifier_dropout_rate,
                    logits=logits,
                )
            elif self.classify_all:
                self.categorical_classifiers[attribute_] = Classifier(
                    n_input=self.n_genes,
                    n_labels=len(unique_categories),
                    n_hidden=classifier_nn_width,
                    n_layers=classifier_nn_depth,
                    dropout_rate=classifier_dropout_rate,
                    logits=logits,
                )

    def _get_inference_input(self, tensors: Dict, **kwargs):
        x = tensors[self.x_loc]  # batch_size, n_genes
        sample_indices = tensors[REGISTRY_KEYS.INDICES_KEY].long().ravel()

        classification = self.classify(x)

        categorical_attribute_dict = {}
        for attribute_ in self.categorical_attributes_map:
            categorical_attribute_dict[attribute_] = tensors[attribute_].view(
                -1,
            )
            if self.categorical_attributes_missing[attribute_] is not None:
                idx_ = categorical_attribute_dict[attribute_] == self.categorical_attributes_missing[attribute_]
                categorical_attribute_dict[attribute_][idx_] = classification[attribute_][idx_]

        ordered_attribute_dict = {}
        for attribute_ in self.ordered_attributes_map:
            ordered_attribute_dict[attribute_] = tensors[attribute_]

        input_dict = {
            "genes": x,
            "sample_indices": sample_indices,
            "categorical_attribute_dict": categorical_attribute_dict,
            "ordered_attribute_dict": ordered_attribute_dict,
        }
        return input_dict

    @auto_move_data
    def classify(
        self,
        genes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run classification.

        Parameters
        ----------
        genes
            Gene expression used for classification.

        Returns
        -------
        Classification output, probability for each ordered attribute and the regression value
        for ordered attributes
        """
        classification = {}
        for attribute_, classifier_ in self.categorical_classifiers.items():
            classification[attribute_] = classifier_(genes)

        for attribute_, regressor_ in self.ordered_regressors.items():
            classification[attribute_] = regressor_(genes)

        return classification

    @auto_move_data
    def _classification_loss(self, tensors: Dict[str, torch.Tensor]):
        """Get module classification loss."""
        x = tensors[self.x_loc]  # batch_size, n_genes

        classification_loss = torch.tensor([0.0]).to(self.device)
        classification = self.classify(x)

        for attribute_ in self.categorical_classifiers:
            attribute_vals = tensors[attribute_].view(-1).long()
            if self.categorical_attributes_missing[attribute_] is not None:
                idx_ = (
                    attribute_vals
                    == self.categorical_attributes_map[attribute_][self.categorical_attributes_missing[attribute_]]
                )
                assignment = Categorical(classification[attribute_][idx_, :])
                loss_fn = nn.CrossEntropyLoss(
                    ignore_index=self.categorical_attributes_map[attribute_][
                        self.categorical_attributes_missing[attribute_]
                    ]
                )
                classification_loss += (
                    loss_fn(
                        classification[attribute_],
                        attribute_vals,
                    )
                    + assignment.entropy().mean()
                )
            else:
                classification_loss += self.classification_loss_fn(
                    classification[attribute_],
                    attribute_vals,
                )

        for attribute_, len_ in self.ordered_attributes_map.items():
            if attribute_ in classification:
                if len_ > 1:
                    if self.loss_regression == "mse":
                        classification_loss += self.regression_loss_fn(
                            classification[attribute_], tensors[attribute_].float()
                        ) + self.classification_penalty * self.mm_regression_loss_fn(
                            classification[attribute_], tensors[attribute_].gt(0).float()
                        )
                    else:
                        classification_loss += self.regression_loss_fn(
                            classification[attribute_][0],
                            tensors[attribute_].float(),
                            classification[attribute_][1],
                        )
                else:
                    classification_loss += self.regression_loss_fn(
                        classification[attribute_][0],
                        tensors[attribute_].float(),
                        classification[attribute_][1],
                    )

        return classification_loss

    @auto_move_data
    def loss(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[Literal["latent_unknown_attributes"], torch.Tensor],
        generative_outputs: Dict[Literal["distribution", "means", "variances"], torch.Tensor],
    ) -> Dict[str, float]:
        """Compute the loss.

        Parameters
        ----------
        tensors
            Considered model inputs.
        inference_outputs
            Inference step outputs.
        generative_outputs
            Generative step outputs.

        Returns
        -------
        The loss elements.
        """
        losses = super().loss(
            tensors=tensors,
            inference_outputs=inference_outputs,
            generative_outputs=generative_outputs,
        )

        losses[LOSS_KEYS.CLASSIFICATION] = self._classification_loss(tensors)

        return losses

    @torch.no_grad()
    def _classification_accuracy(self, tensors):
        """Get module classification accuracy."""
        accuracy_dict = {}
        accuracy_val = 0
        r2_dict = {}
        r2_val = 0
        mse_dict = {}
        mse_val = 0

        x = tensors[self.x_loc]  # .detach().cpu().numpy()  # batch_size, n_genes

        classification = self.classify(x)

        for attribute_ in self.categorical_classifiers:
            attribute_vals_pred = classification[attribute_].argmax(dim=-1).cpu().numpy()
            attribute_vals = tensors[attribute_].view(-1).long().cpu().numpy()
            if self.categorical_attributes_missing[attribute_] is not None:
                idx_ = (
                    attribute_vals
                    != self.categorical_attributes_map[attribute_][self.categorical_attributes_missing[attribute_]]
                )
                if idx_.sum() > 0:
                    accuracy_dict[attribute_] = np.mean(attribute_vals_pred[idx_] == attribute_vals[idx_])
                    accuracy_val += accuracy_dict[attribute_]
            else:
                accuracy_dict[attribute_] = np.mean(attribute_vals_pred == attribute_vals)
                accuracy_val += accuracy_dict[attribute_]

        for attribute_ in self.ordered_regressors:
            attribute_vals_pred = (
                classification[attribute_] if self.regression_loss == "mse" else classification[attribute_][0]
            )
            attribute_vals = tensors[attribute_].cpu().numpy()
            attribute_vals_pred = (
                torch.nan_to_num(attribute_vals_pred, nan=0, neginf=0, posinf=100).detach().cpu().numpy()
            )

            r2_dict[attribute_] = r2_score(attribute_vals, attribute_vals_pred) if attribute_vals.shape[0] > 2 else 0
            mse_dict[attribute_] = mean_squared_error(attribute_vals, attribute_vals_pred)

            r2_val += r2_dict[attribute_]
            mse_val += mse_dict[attribute_]

        accuracy_mean = 0
        r2_mean = 0
        mse_mean = 0
        if len(accuracy_dict):
            accuracy_mean = accuracy_val / len(accuracy_dict)
        if len(r2_dict):
            r2_mean = r2_val / len(r2_dict)
            mse_mean = mse_val / len(mse_dict)

        return accuracy_dict, accuracy_mean, r2_dict, r2_mean, mse_dict, mse_mean
