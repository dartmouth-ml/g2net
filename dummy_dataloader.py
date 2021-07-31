from pl_bolts.datasets import RandomDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

dataloader = DataLoader(RandomDataset((3, 128, 128), num_samples=64),
                        batch_size=64,
                        shuffle=True)

class DummyModule(LightningDataModule):
    def __init__(self):
        super().__init__()
    
    def train_dataloader(self):
        return DataLoader(RandomDataset((3, 128, 128), num_samples=64), batch_size=64, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(RandomDataset((3, 128, 128), num_samples=64), batch_size=64, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(RandomDataset((3, 128, 128), num_samples=64), batch_size=64, shuffle=True)
    