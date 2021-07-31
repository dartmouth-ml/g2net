from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch

class RandomDataset(Dataset):
    """
    Generate a dummy dataset

    Example:
        >>> from pl_bolts.datasets import RandomDataset
        >>> from torch.utils.data import DataLoader
        >>> ds = RandomDataset(10)
        >>> dl = DataLoader(ds, batch_size=7)
    """

    def __init__(self, size: int, num_samples: int = 250):
        """
        Args:
            size: tuple
            num_samples: number of samples
        """
        self.len = num_samples
        self.data = [torch.randn(size) for _ in range(num_samples)]

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return self.len

dataloader = DataLoader(RandomDataset((3, 128, 128), num_samples=64),
                        batch_size=64,
                        shuffle=True)

class DummyModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
    
    def train_dataloader(self):
        return DataLoader(RandomDataset(size=(3, 128, 128), num_samples=64), batch_size=64, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(RandomDataset(size=(3, 128, 128), num_samples=64), batch_size=64, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(RandomDataset(size=(3, 128, 128), num_samples=64), batch_size=64, shuffle=True)
    