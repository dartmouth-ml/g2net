import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from spectrogram import make_spectrogram
import einops

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize

from pytorch_lightning import LightningDataModule

class SpectrogramDataset(Dataset):
    def __init__(self, data_path: Path, labels_df: pd.DataFrame, transforms=None):
        super().__init__()
        self.time_series_path = data_path

        self.file_names = labels_df['id'].tolist()
        self.labels = np.array(labels_df['target'].tolist())

        self.transforms = transforms
        self.file_ext = '.npy'

    def __getitem__(self, idx):
        """
        spectrogram -- (currently) 9 x 128 color image
        label -- 0 or 1
        """
        file_name = self.file_names[idx]
        full_path = self.convert_to_full_path(file_name)
        time_series_data = np.load(full_path).astype(np.float32)

        spectrogram = make_spectrogram(time_series_data)
        spectrogram = einops.rearrange(spectrogram, 't c f -> c f t')
        

        label = self.labels[idx]
        spectrogram = self.transforms(spectrogram)

        self.ts_transforms = Compose([ToTensor()])

        time_series_data = self.ts_transforms(time_series_data)

        return spectrogram, label, file_name, time_series_data

    def convert_to_full_path(self, file_name):
        full_path = self.time_series_path.joinpath(*[s for s in file_name[:3]], file_name).with_suffix(self.file_ext)
        assert full_path.is_file(), full_path

        return full_path

    def __len__(self):
        return len(self.file_names)

class G2NetDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if not self.config.validation_labels_path.is_file():
            self.split_train_val()
        
        self.transforms = self.get_transforms()
        self.datasets = self.get_datasets()
    
    def split_train_val(self):
        all_labels_path = self.config.all_labels_path
        all_df = pd.read_csv(all_labels_path)

        labels_x = all_df.iloc[:, 0]
        labels_y = all_df.iloc[:, 1]
        m = all_df.shape[0]
        train_size = int(m - (m * self.config.val_ratio))
        filenames_train, filenames_val, labels_train, labels_val = train_test_split(labels_x, 
                                                                                    labels_y,
                                                                                    train_size=train_size,
                                                                                    shuffle=True)
        train_df = pd.concat([filenames_train, labels_train], axis=1)
        val_df = pd.concat([filenames_val, labels_val], axis=1)

        train_df.to_csv(self.config.training_labels_path)
        val_df.to_csv(self.config.validation_labels_path)


    def get_transforms(self):
        train_transforms = Compose([ToTensor()])
        val_transforms = Compose([ToTensor()])

        return {'train': train_transforms, 'val': val_transforms}

    def get_datasets(self):
        train_df = pd.read_csv(self.config.training_labels_path)
        val_df = pd.read_csv(self.config.validation_labels_path)
        test_df = pd.read_csv(self.config.test_labels_path)

        train_dset = SpectrogramDataset(self.config.data_path.joinpath('train'),
                                        labels_df=train_df,
                                        transforms=self.transforms['train'])

        val_dset = SpectrogramDataset(self.config.data_path.joinpath('train'),
                                      labels_df=val_df,
                                      transforms=self.transforms['val'])
        
        test_dset = SpectrogramDataset(self.config.data_path.joinpath('test'),
                                       labels_df=test_df,
                                       transforms=self.transforms['val'])

        return {'train': train_dset, 'val': val_dset, 'test': test_dset}

    def train_dataloader(self):
       return DataLoader(dataset=self.datasets['train'],
                         batch_size=self.config.batch_size,
                         shuffle=True,
                         num_workers=self.config.num_workers)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.datasets['val'],
                         batch_size=self.config.batch_size,
                         shuffle=False,
                         num_workers=self.config.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(dataset=self.datasets['test'],
                         batch_size=self.config.batch_size,
                         shuffle=False,
                         num_workers=self.config.num_workers)
