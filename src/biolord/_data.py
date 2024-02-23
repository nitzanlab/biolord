from typing import Optional

from scvi.data import AnnDataManager
from scvi.dataloaders import DataSplitter


class AnnDataSplitter(DataSplitter):
    """Creates data loaders ``train_set``, ``validation_set``, ``test_set`` using given indices."""

    def __init__(
        self,
        adata_manager: AnnDataManager,
        train_indices,
        valid_indices,
        test_indices,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__(adata_manager=adata_manager, pin_memory=pin_memory)
        self.data_loader_kwargs = kwargs
        self.train_idx = train_indices
        self.val_idx = valid_indices
        self.test_idx = test_indices

    def setup(self, stage: Optional[str] = None):
        """Over-ride parent's setup to preserve split idx."""
        return
