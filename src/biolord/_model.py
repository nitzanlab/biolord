import io
import itertools
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rich
import torch
from anndata import AnnData
from pytorch_lightning.callbacks import ModelCheckpoint
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalObsField,
    LayerField,
    NumericalObsField,
    ObsmField,
)
from scvi.dataloaders import DataSplitter
from scvi.model.base import BaseModelClass
from scvi.train import TrainRunner
from scvi.utils import setup_anndata_dsp
from tqdm import tqdm

from ._data import AnnDataSplitter
from ._module import BiolordClassifyModule, BiolordModule
from ._train import biolordClassifyTrainingPlan, biolordTrainingPlan
from ._utils import repeat_n

logger = logging.getLogger(__name__)
logger.propagate = False
logging_dir = "./biolord_log/"

__all__ = ["Biolord"]


class Biolord(BaseModelClass):
    """The biolord model class.

    Parameters
    ----------
    adata
        Annotated data object.
    model_name
        Name of the model.
    module_params
        Hyperparameters for the model's module initialization, e.g, :class:`~biolord.BiolordModule` or
        :class:`~biolord.BiolordClassifyModule`.
    n_latent
        Number of latent dimensions used for the latent embedding.
    train_classifiers
        Whether to activate a :class:`~biolord.BiolordClassifyModule`.
    split_key
        Key in :attr:`anndata.AnnData.obs` used to split the data between train, test and validation.
    train_split
        Value in :attr:`anndata.AnnData.obs` ``['{split_key}']`` marking the train set.
    valid_split
        Value in :attr:`anndata.AnnData.obs` ``['{split_key}']`` marking the validation set.
    test_split
        Value in :attr:`anndata.AnnData.obs` ``['{split_key}']`` marking the test set.

    Examples
    --------
    .. code-block:: python

        import scanpy as sc
        import biolord

        adata = sc.read(...)
        biolord.Biolord.setup_anndata(
            adata, ordered_attributes_keys=["time"], categorical_attributes_keys=["cell_type"]
        )
        model = biolord.Biolord(adata, n_latent=256, split_key="split")
        model.train(max_epochs=200, batch_size=256)
    """

    def __init__(
        self,
        adata: AnnData,
        model_name: Optional[str] = None,
        module_params: Dict[str, Any] = None,
        n_latent: int = 128,
        train_classifiers: bool = False,
        split_key: Optional[str] = None,
        train_split: str = "train",
        valid_split: str = "test",
        test_split: str = "ood",
    ):
        super().__init__(adata)

        self.categorical_attributes_map = {}
        self.ordered_attributes_map = {}
        self.retrieval_attribute_dict = {}
        self.categorical_attributes_missing = self.registry_["setup_args"]["categorical_attributes_missing"]
        self.x_loc = None

        self._set_attributes_maps()

        self.n_latent = n_latent
        self.n_genes = adata.n_vars

        self.split_key = split_key
        self.scores = {}

        self._module = None
        self._training_plan = None
        self._data_splitter = None

        train_indices, valid_indices, test_indices = None, None, None
        if split_key is not None:
            train_indices = np.where(adata.obs.loc[:, split_key] == train_split)[0]
            valid_indices = np.where(adata.obs.loc[:, split_key] == valid_split)[0]
            test_indices = np.where(adata.obs.loc[:, split_key] == test_split)[0]

        self.train_indices = train_indices
        self.valid_indices = valid_indices
        self.test_indices = test_indices
        self.n_samples = adata.n_obs
        self.train_classifiers = train_classifiers

        module_params = module_params if isinstance(module_params, Dict) else {}

        if self.train_classifiers:
            self.module = BiolordClassifyModule(
                n_genes=self.n_genes,
                n_samples=self.n_samples,
                x_loc=self.x_loc,
                categorical_attributes_map=self.categorical_attributes_map,
                ordered_attributes_map=self.ordered_attributes_map,
                categorical_attributes_missing=self.categorical_attributes_missing,
                n_latent=self.n_latent,
                **module_params,
            ).float()
        else:
            self.module = BiolordModule(
                n_genes=self.n_genes,
                n_samples=self.n_samples,
                x_loc=self.x_loc,
                ordered_attributes_map=self.ordered_attributes_map,
                categorical_attributes_map=self.categorical_attributes_map,
                n_latent=self.n_latent,
                **module_params,
            ).float()

        self._model_summary_string = self.__class__.__name__
        self._model_name = model_name
        self.init_params_ = self._get_init_params(locals())
        self.epoch_history = None

    def _set_attributes_maps(self):
        """Set attributes' maps."""
        for attribute_ in self.registry_["setup_args"]["categorical_attributes_keys"]:
            self.categorical_attributes_map[attribute_] = {
                c: i
                for i, c in enumerate(
                    self.registry_["field_registries"][attribute_]["state_registry"]["categorical_mapping"]
                )
            }
        for attribute_ in self.registry_["setup_args"]["ordered_attributes_keys"]:
            # validata obs
            if attribute_ in self.adata.obs:
                self.ordered_attributes_map[attribute_] = 1
            elif attribute_ in self.adata.obsm:
                self.ordered_attributes_map[attribute_] = self.adata.obsm[attribute_].shape[1]
            else:
                raise KeyError(f"class {attribute_} not found in `adata.obs` or `adata.obsm`.")

        if self.registry_["setup_args"]["retrieval_attribute_key"] is not None:
            self.retrieval_attribute_dict = {
                "retrieval_attribute_key": len(
                    np.unique(self.adata.obs[self.registry_["setup_args"]["retrieval_attribute_key"]])
                )
            }

        self.x_loc = self.registry_["setup_args"]["FIELD"].attr_name

    @property
    def training_plan(self):
        """The model's training plan."""
        return self._training_plan

    @training_plan.setter
    def training_plan(self, plan):
        self._training_plan = plan

    @property
    def data_splitter(self):
        """Data splitter."""
        return self._data_splitter

    @data_splitter.setter
    def data_splitter(self, data_splitter):
        self._data_splitter = data_splitter

    @property
    def module(self) -> BiolordModule:
        """Model's module."""
        return self._module

    @module.setter
    def module(self, module: BiolordModule):
        self._module = module

    @property
    def model_name(self) -> str:
        """Model's name."""
        return self._model_name

    @model_name.setter
    def model_name(self, model_name: str):
        self._model_name = model_name

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        ordered_attributes_keys: Optional[List[str]] = None,
        categorical_attributes_keys: Optional[List[str]] = None,
        categorical_attributes_missing: Optional[Dict[str, str]] = None,
        retrieval_attribute_key: Optional[str] = None,
        layer: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Setup function.

        Parameters
        ----------
        adata
            Annotated data object.
        ordered_attributes_keys
            Valid :attr:`anndata.AnnData.obs` or :attr:`anndata.AnnData.obsm` keys for the ordered attributes.
        categorical_attributes_keys
            Valid :attr:`anndata.AnnData.obs` keys for the categorical attributes.
        categorical_attributes_missing
            Categories representing missing labels. Only used if ``train_classifiers=True``.
        retrieval_attribute_key
            Valid :attr:`anndata.AnnData.obs` key for an attribute to evaluate retrieval performance over.
        layer
            Expression layer in :attr:`anndata.AnnData.layers` to use. If :obj:`None`, use :attr:`anndata.AnnData.X`.
        kwargs
            Keyword arguments for :meth:`~scvi.data.AnnDataManager.register_fields`.

        Returns
        -------
        Nothing, just sets up ``adata``.
        """
        if layer is not None:
            if layer not in adata.layers:
                raise KeyError(f"{layer} is not a valid key in `adata.layers`.")
            logger.info(f"Using data from adata.layers[{layer!r}]")
            FIELD = LayerField(
                registry_key="layers",
                layer=layer,
                is_count_data=True,
            )
        else:
            logger.info("Using data from `adata.X`.")
            FIELD = LayerField(registry_key="X", layer=None, is_count_data=False)

        ordered_attributes_keys = ordered_attributes_keys if isinstance(ordered_attributes_keys, List) else []

        categorical_attributes_keys = (
            categorical_attributes_keys if isinstance(categorical_attributes_keys, List) else []
        )

        if categorical_attributes_missing is not None:
            for attribute_, val_ in categorical_attributes_missing.items():
                if val_ is not None:
                    adata.obs[attribute_] = adata.obs[attribute_].astype("category")
                    cats = adata.obs[attribute_].cat.categories
                    idx_ = cats.isin([val_])
                    ordered = list(cats[~idx_]) + [val_]

                    adata.obs[attribute_] = adata.obs[attribute_].cat.reorder_categories(ordered)

        # set retrieval class
        retrieval_attribute_dict = {}
        if retrieval_attribute_key is not None:
            retrieval_attribute_dict = {"retrieval_attribute_key": len(np.unique(adata.obs[retrieval_attribute_key]))}

        # set ordered classes
        ordered_attributes_obs = []
        ordered_attributes_obsm = []
        for attribute_ in ordered_attributes_keys:
            # validata obs
            if attribute_ in adata.obs:
                ordered_attributes_obs.append(attribute_)
            elif attribute_ in adata.obsm:
                ordered_attributes_obsm.append(attribute_)
            else:
                raise KeyError(f"class {attribute_} not found in `adata.obs` or `adata.obsm`.")

        setup_method_args = cls._get_setup_method_args(**locals())
        adata.obs["_indices"] = np.arange(adata.n_obs)
        anndata_fields = (
            [
                FIELD,
                NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
            ]
            + [
                CategoricalObsField(registry_key=attribute_, attr_key=attribute_)
                for attribute_ in categorical_attributes_keys
            ]
            + [
                NumericalObsField(
                    attribute_,
                    attribute_,
                )
                for attribute_ in ordered_attributes_obs
            ]
            + [
                ObsmField(
                    attribute_,
                    attribute_,
                    is_count_data=False,
                    correct_data_format=True,
                )
                for attribute_ in ordered_attributes_obsm
            ]
        )
        if retrieval_attribute_key is not None:
            anndata_fields += [
                CategoricalObsField(
                    registry_key=retrieval_attribute_key,
                    attr_key=retrieval_attribute_key,
                )
            ]

        adata_manager = AnnDataManager(
            fields=anndata_fields,
            setup_method_args=setup_method_args,
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    def get_latent_representation_adata(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = 512,
        nullify_attribute: Optional[List[str]] = None,
    ) -> Tuple[AnnData, AnnData]:
        """Return the unknown attributes latent space and full latent variable.

        Parameters
        ----------
        adata
            Annotated data object.
        indices
            Optional indices.
        batch_size
            Batch size to use.
        nullify_attribute
            Attribute to nullify in the latent space.

        Returns
        -------
        Two :class:`~anndata.AnnData` objects providing the unknown attributes latent space and
        the concatenated decomposed latent respectively.
        """
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        nullify_attribute = [] if nullify_attribute is None else nullify_attribute
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size, shuffle=False)

        latent_unknown_attributes = []
        latent = []
        for tensors in scdl:
            inference_inputs = self.module.get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs, nullify_attribute=nullify_attribute)
            latent_unknown_attributes += [outputs["latent_unknown_attributes"].cpu().numpy()]
            latent += [outputs["latent"].cpu().numpy()]

        latent_unknown_attributes_adata = AnnData(
            X=np.concatenate(latent_unknown_attributes, axis=0), obs=adata[indices].obs.copy()
        )
        latent_unknown_attributes_adata.obs_names = adata[indices].obs_names

        latent_adata = AnnData(X=np.concatenate(latent, axis=0), obs=adata[indices].obs.copy())
        latent_adata.obs_names = adata[indices].obs_names

        return latent_unknown_attributes_adata, latent_adata

    @torch.no_grad()
    def get_dataset(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Processes :class:`~anndata.AnnData` object into valid input tensors for the model.

        Parameters
        ----------
        adata
            Annotated data object.
        indices
            Optional indices.

        Returns
        -------
        A dictionary of tensors which can be passed as input to the model.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=len(indices), shuffle=False)
        return list(scdl)[0]

    @torch.no_grad()
    def predict(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = 512,
        nullify_attribute: Optional[List[str]] = None,
    ) -> Tuple[AnnData, AnnData]:
        """The model's gene expression prediction for a given :class:`~anndata.AnnData` object.

        Parameters
        ----------
        adata
            Annotated data object.
        indices
            Optional indices.
        batch_size
            Batch size to use.
        nullify_attribute
            Attribute to nullify in latent space.

        Returns
        -------
        Two :class:`~anndata.AnnData` objects representing the model's prediction of the expression mean and variance respectively.
        """
        nullify_attribute = [] if nullify_attribute is None else nullify_attribute
        self.module.eval()

        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size, shuffle=False)
        mus = []
        stds = []
        for tensors in scdl:
            _mus, _stds = self.module.get_expression(tensors, nullify_attribute=nullify_attribute)
            _mus = _mus if _mus.ndim > 1 else _mus[None, :]
            _stds = _stds if _stds.ndim > 1 else _stds[None, :]
            mus.append(_mus.detach().cpu().numpy())
            stds.append(_stds.detach().cpu().numpy())

        pred_adata_mean = AnnData(X=np.concatenate(mus, axis=0), obs=adata.obs.copy())
        pred_adata_var = AnnData(X=np.concatenate(stds, axis=0), obs=adata.obs.copy())

        pred_adata_mean.obs_names = adata.obs_names
        pred_adata_var.obs_names = adata.obs_names

        pred_adata_mean.var_names = adata.var_names
        pred_adata_var.var_names = adata.var_names

        return pred_adata_mean, pred_adata_var

    @torch.no_grad()
    def get_ordered_attribute_embedding(
        self,
        attribute_key: str,
        vals: Optional[Union[float, str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Compute embedding of an ordered attribute.

        Parameters
        ----------
        attribute_key
            The key of the desired attribute.
        vals
            Values of interest.

        Returns
        -------
        Array of the attribute's embedding.
        """
        self.module.eval()
        vals = vals if vals is not None else 1.0
        if isinstance(vals, float):
            batch = torch.tensor([vals]).to(self.device).float()
        elif isinstance(vals, np.ndarray):
            batch = torch.tensor(vals).to(self.device).float()
        else:
            batch = vals

        return self.module.ordered_networks[attribute_key](batch).detach().cpu().numpy()

    @torch.no_grad()
    def get_categorical_attribute_embeddings(
        self, attribute_key: str, attribute_category: Optional[str] = None
    ) -> np.ndarray:
        """Compute embedding of a categorical attribute.

        Parameters
        ----------
        attribute_key
            The key of the desired attribute.
        attribute_category
            A specific category for embedding computation.

        Returns
        -------
        Array of the attribute's embedding.
        """
        if attribute_category is None:
            cat_ids = torch.arange(len(self.categorical_attributes_map[attribute_key]), device=self.device)
        else:
            cat_ids = torch.LongTensor([self.categorical_attributes_map[attribute_key][attribute_category]]).to(
                self.device
            )
        embeddings = self.module.categorical_embeddings[attribute_key](cat_ids).detach().cpu().numpy()

        return embeddings

    def save(
        self,
        dir_path: Optional[str] = None,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_save_kwargs: Any,
    ) -> None:
        """Save the model.

        Parameters
        ----------
        dir_path
            Directory where to save the model. If :obj:`None`, it will be determined automatically.
        overwrite
            Whether to overwrite an existing model.
        save_anndata
            Whether to also save :class:`~anndata.AnnData`.
        anndata_save_kwargs
            Keyword arguments :meth:`scvi.model.base.BaseModelClass.save`.

        Returns
        -------
        Nothing, just saves the model.
        """
        if dir_path is None:
            dir_path = (
                f"./{self.__class__.__name__}_model/"
                if self.model_name is None
                else f"./{self.__class__.__name__}_{self.model_name}_model/"
            )
        super().save(
            dir_path=dir_path,
            overwrite=overwrite,
            save_anndata=save_anndata,
            **anndata_save_kwargs,
        )

        if isinstance(self.training_plan.epoch_history, dict):
            self.epoch_history = pd.DataFrame().from_dict(self.training_plan.epoch_history)
            self.epoch_history.to_csv(os.path.join(dir_path, "history.csv"), index=False)

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata: Optional[AnnData] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        **kwargs: Any,
    ) -> "Biolord":
        """Load a saved model.

        Parameters
        ----------
        dir_path
            Directory where the model is saved.
        adata
            AnnData organized in the same way as data used to train model.
        use_gpu
            Whether to load the model on GPU.
        kwargs
            Keyword arguments for :meth:`scvi`

        Returns
        -------
        The saved model.
        """
        model = super().load(dir_path, adata, use_gpu=use_gpu, **kwargs)

        Biolord.categorical_attributes_map = model.categorical_attributes_map
        Biolord.ordered_attributes_map = model.ordered_attributes_map

        fname = os.path.join(dir_path, "history.csv")
        if os.path.isfile(fname):
            model.epoch_history = pd.read_csv(fname)
        else:
            logger.warning(f"The history file `{fname}` was not found")

        return model

    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[bool] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        plan_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        **trainer_kwargs: Any,
    ) -> None:
        """Train the :class:`~biolord.Biolord` model.

        Parameters
        ----------
        max_epochs
            Maximum number of epochs for training.
        use_gpu
            Whether to use GPU if available.
        train_size
            Fraction of training data in the case of randomly splitting dataset to train/validation
            if :attr:`split_key` is not set in model's constructor.
        validation_size
            Fraction of validation data in the case of randomly splitting dataset to train/validation
            if :attr:`split_key` is not set in model's constructor.
        batch_size
            Size of mini-batches for training.
        early_stopping
            If `True`, early stopping will be used during training on validation dataset.
        plan_kwargs
            Keyword arguments for :class:`~scvi.train.TrainingPlan`.
        trainer_kwargs
            Keyword arguments for :class:`~scvi.train.TrainRunner`.

        Returns
        -------
        Nothing, just trains the :class:`~biolord.Biolord` model.
        """
        plan_kwargs = plan_kwargs if plan_kwargs is not None else {}
        if self.train_classifiers:
            self.training_plan = biolordClassifyTrainingPlan(
                module=self.module,
                **plan_kwargs,
            )
        else:
            self.training_plan = biolordTrainingPlan(
                module=self.module,
                **plan_kwargs,
            )

        monitor = trainer_kwargs.pop("monitor", "val_biolord_metric")
        save_ckpt_every_n_epoch = trainer_kwargs.pop("save_ckpt_every_n_epoch", 20)
        enable_checkpointing = trainer_kwargs.pop("enable_checkpointing", True)

        trainer_kwargs["callbacks"] = [] if "callbacks" not in trainer_kwargs.keys() else trainer_kwargs["callbacks"]
        if enable_checkpointing:
            checkpointing_callback = ModelCheckpoint(monitor=monitor, every_n_epochs=save_ckpt_every_n_epoch)
            trainer_kwargs["callbacks"] += [checkpointing_callback]

        num_workers = trainer_kwargs.pop("num_workers", 0)
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        manual_splitting = (
            (self.valid_indices is not None) and (self.train_indices is not None) and (self.test_indices is not None)
        )

        if manual_splitting:
            self.data_splitter = AnnDataSplitter(
                self.adata_manager,
                train_indices=self.train_indices,
                valid_indices=self.valid_indices,
                test_indices=self.test_indices,
                batch_size=batch_size,
                use_gpu=use_gpu,
                num_workers=num_workers,
            )
        else:
            self.data_splitter = DataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                use_gpu=use_gpu,
                num_workers=num_workers,
            )
        es = "early_stopping"
        trainer_kwargs[es] = early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]

        trainer_kwargs["check_val_every_n_epoch"] = trainer_kwargs.get("check_val_every_n_epoch", 1)
        trainer_kwargs["early_stopping_patience"] = trainer_kwargs.get("early_stopping_patience", 20)

        root_dir = logging_dir
        root_dir = (
            os.path.join(root_dir, f"{self.__class__.__name__}/")
            if self.model_name is None
            else os.path.join(root_dir, f"{self.model_name}_{self.__class__.__name__}/")
        )

        trainer_kwargs["default_root_dir"] = trainer_kwargs.pop("default_root_dir", root_dir)

        runner = TrainRunner(
            self,
            training_plan=self.training_plan,
            data_splitter=self.data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            early_stopping_monitor=monitor,
            early_stopping_mode="max",
            enable_checkpointing=enable_checkpointing,
            **trainer_kwargs,
        )

        return runner()

    @torch.no_grad()
    def evaluate_retrieval(
        self,
        batch_size: int = None,
        eval_set: Literal["test", "validation"] = "test",
    ) -> float:
        """Returns the accuracy of the retrieval task over the pre-defined `retrieval_class`.

        Parameters
        ----------
        batch_size
            Batch size to use.
        eval_set
            Evaluation dataset.

        Returns
        -------
        Retrieval accuracy over the evaluation dataset.
        """
        k = 1
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        batch_size = batch_size if batch_size is not None else self.adata.n_obs

        latent_unknown_attributes_train_adata, _ = self.get_latent_representation_adata(
            adata=self.adata, indices=self.train_indices, batch_size=batch_size
        )

        if eval_set == "test":
            eval_indices = self.test_indices
        elif eval_set == "validation":
            eval_indices = self.valid_indices
        else:
            raise RuntimeError(f"Supports `eval_type` `test` or `validation` but {eval_set} was provided.")

        latent_unknown_attributes_test_adata, _ = self.get_latent_representation_adata(
            adata=self.adata, indices=eval_indices
        )

        return self._retrieval_accuracy(
            latent_unknown_attributes_test_adata.obs[REGISTRY_KEYS.RETRIEVAL_CLASS],
            latent_unknown_attributes_test_adata.X,
            latent_unknown_attributes_train_adata.obs[REGISTRY_KEYS.RETRIEVAL_CLASS],
            latent_unknown_attributes_train_adata.X,
            k=k,
        )

    @torch.no_grad()
    def _retrieval_accuracy(
        self,
        retrieval_class,
        latent_unknown_attributes,
        retrieval_attribute_train,
        latent_unknown_attributes_train,
        k=1,
    ) -> float:
        """Retrieval accuracy score."""
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=k).fit(latent_unknown_attributes_train)
        _, ind_ = nbrs.kneighbors(latent_unknown_attributes)

        tot_correct = 0.0
        for i_eval, i_train in enumerate(ind_):
            is_equal = [(retrieval_attribute_train[i_train[i]] == retrieval_class[i_eval]) for i in range(k)]
            is_equal = np.sum(is_equal) / k
            tot_correct = tot_correct + is_equal

        return tot_correct / len(ind_)

    @torch.no_grad()
    def _compute_prediction(
        self,
        adata,
        dataset_source,
        target_attributes,
    ) -> Tuple[Dict[Tuple[Any], Any], Any]:
        """Expression prediction over given inputs.

        Parameters
        ----------
        adata
            An annotated data object containing possible values of the `target_attributes`.
        dataset_source
            Dataset of cells to "shift", make predictions over.
        target_attributes
            Attributes to make predictions over.

        Returns
        -------
        The prediction dict for each attribute value and the original expression prediction.
        """
        keys = list(
            itertools.product(
                *[list(self.categorical_attributes_map[attribute_].keys()) for attribute_ in target_attributes]
            )
        )
        layer = "X" if "X" in dataset_source else "layers"

        pred_original, _ = self.module.get_expression(dataset_source)

        classes_dataset = {}
        predictions_dict = {}

        for attribute_ in target_attributes:
            categories_index = pd.Index(adata.obs[attribute_].values, dtype="category")
            classes_dataset[attribute_] = {}
            for key_, _ in tqdm(zip(*np.unique(categories_index.values, return_counts=True))):
                bool_category = categories_index.get_loc(key_)

                adata_cur = adata[bool_category, :].copy()
                dataset = self.get_dataset(adata_cur)

                classes_dataset[attribute_][key_] = dataset[attribute_][0, :]

        for key_ in keys:
            dataset_comb = {}
            n_obs = dataset_source[layer].size(0)
            for key_dataset in dataset_source:
                dataset_comb[key_dataset] = dataset_source[key_dataset].to(self.device)

            for ci, attribute_ in enumerate(target_attributes):
                dataset_comb[attribute_] = repeat_n(classes_dataset[attribute_][key_[ci]], n_obs)

            pred, _ = self.module.get_expression(dataset_comb)
            predictions_dict[key_] = pred

        return predictions_dict, pred_original

    def compute_prediction_adata(
        self,
        adata: AnnData,
        adata_source: AnnData,
        target_attributes: List[str],
        add_attributes: Optional[List[str]] = None,
    ) -> AnnData:
        """Expression prediction over given inputs.

        Parameters
        ----------
        adata
            Annotated data object containing possible values of the ``target_attributes``.
        adata_source
            Annotated data object we wish to make predictions over, e.g., change their ``target_attributes``.
        target_attributes
            Attributes to make predictions over.
        add_attributes
            Additional attributes to add to :attr:`anndata.AnnData.obs` from the original adata
            to the  prediction adata object.

        Returns
        -------
        Annotated data object containing predictions of the cells in all combinations of the ``target_attributes``.
        """
        dataset_source = self.get_dataset(adata_source)

        predictions_dict, _ = self._compute_prediction(
            adata=adata, dataset_source=dataset_source, target_attributes=target_attributes
        )

        preds_ = np.concatenate([val.cpu() for key, val in predictions_dict.items()])
        adata_preds = AnnData(X=preds_, dtype=preds_.dtype)
        for attribute_ in target_attributes:
            adata_preds.obs[attribute_] = "Source"

        start = 0
        obs_names_tmp = adata_preds.obs_names.values
        for key_, vals_ in predictions_dict.items():
            for ci, _ in enumerate(target_attributes):
                adata_preds.obs.iloc[start : start + vals_.shape[0], ci] = key_[ci]
            obs_names_tmp[start : start + vals_.shape[0]] = [
                obs_name + "_" + "_".join([str(k) for k in key_]) for obs_name in adata_source.obs_names
            ]
            start += vals_.shape[0]

        adata_preds.obs_names = obs_names_tmp
        for attribute_ in target_attributes:
            adata_preds.obs[attribute_] = adata_preds.obs[attribute_].astype("category")

        if add_attributes is not None:
            for attribute_ in add_attributes:
                start = 0
                adata_preds.obs[attribute_] = np.nan
                for _ in range(int(adata_preds.shape[0] / adata_source.shape[0])):
                    adata_preds.obs.iloc[start : start + adata_source.shape[0], -1] = adata_source.obs[attribute_]
                    start += adata_source.shape[0]

                adata_preds.obs[attribute_] = adata_preds.obs[attribute_].astype(adata_source.obs[attribute_].dtype)
                if f"{attribute_}_colors" in adata_source.uns:
                    adata_preds.uns[f"{attribute_}_colors"] = adata_source.uns[f"{attribute_}_colors"]

        adata_preds.var_names = adata_source.var_names

        return adata_preds

    def __repr__(self) -> str:
        buffer = io.StringIO()
        summary_string = f"{self._model_summary_string} training status: "
        summary_string += "{}".format("[green]Trained[/]" if self.is_trained else "[red]Not trained[/]")
        console = rich.console.Console(file=buffer)
        with console.capture() as capture:
            console.print(summary_string)
        return capture.get()
