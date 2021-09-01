import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class EnsembleDataset(Dataset):
    def __init__(self, submissions):
        super().__init__()
        all_dfs = []

        for submission in submissions:
            sub_df = pd.read_csv(submission).sort_values('id')
            all_dfs.append(sub_df)
        
        self.all_df = pd.concat(all_dfs, axis=1)
    
    def __getitem__(self, idx):
        return self.all_df.iloc[idx, :]
    
class EnsembleDataModule(pl.LightningDataModule):
    def __init__(self, config):
        self.config = config
        self.train_dataset = EnsembleDataset(config.train_submissions)
        self.val_dataset = EnsembleDataset(config.val_submissions)
    
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.config.batch_size,
                          shuffle=True,
                          num_workers=self.config.num_workers)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.config.batch_size,
                          shuffle=False,
                          num_workers=self.config.num_workers)