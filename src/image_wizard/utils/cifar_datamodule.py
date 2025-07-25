import lightning as L
from .download_data import get_cifar10_dataloaders

__all__ = ["CIFAR10DataModule"]


class CIFAR10DataModule(L.LightningDataModule):
    """Lightning DataModule for CIFAR-10."""

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 2,
        subset_fraction: float | None = None,
        subset_size: int | None = None,
        augment: bool = True,
        download: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subset_fraction = subset_fraction
        self.subset_size = subset_size
        self.augment = augment
        self.download = download

    def setup(self, stage: str | None = None):
        """Setup datasets for different stages."""
        # Get dataloaders using your existing function
        self.train_loader, self.val_loader = get_cifar10_dataloaders(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            subset_fraction=self.subset_fraction,
            subset_size=self.subset_size,
            augment=self.augment,
            download=self.download,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.val_loader
