from typing import Optional

from scvi import settings
from scvi.data import AnnDataManager
from scvi.dataloaders import DataSplitter
from scvi.model._utils import parse_use_gpu_arg


class AnnDataSplitter(DataSplitter):
    """Creates data loaders ``train_set``, ``validation_set``, ``test_set`` using given indices."""

    def __init__(
        self,
        adata_manager: AnnDataManager,
        train_indices,
        valid_indices,
        test_indices,
        use_gpu: bool = None,
        **kwargs,
    ):
        super().__init__(adata_manager)
        self.data_loader_kwargs = kwargs
        self.use_gpu = use_gpu
        self.train_idx = train_indices
        self.val_idx = valid_indices
        self.test_idx = test_indices

    def setup(self, stage: Optional[str] = None):
        """Over-ride parent's setup to preserve split idx."""
        accelerator, _, self.device = parse_use_gpu_arg(self.use_gpu, return_device=True)
        self.pin_memory = True if (settings.dl_pin_memory_gpu_training and accelerator == "gpu") else False
