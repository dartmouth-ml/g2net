import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from spectrogram import make_spectrogram

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose

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

        spectrograms = make_spectrogram(time_series_data)
        spectograms = np.stack(spectrograms, axis=0) # 3, n_mels, t

        label = self.labels[idx]
        spectrograms = self.transforms(spectograms)

        return spectrograms, label, file_name

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

        if self.config.val_ratio == 0:
            all_df.to_csv(self.config.training_labels_path)
            return

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
        train_dset = None
        val_dset = None
        test_dset = None

        train_df = None
        val_df = None
        test_df = None

        if self.config.training_labels_path.is_file():
            train_df = pd.read_csv(self.config.training_labels_path)
        
        if self.config.validation_labels_path.is_file():
            val_df = pd.read_csv(self.config.validation_labels_path)
        
        if self.config.test_labels_path.is_file():
            test_df = pd.read_csv(self.config.test_labels_path)

        if train_df is not None:
            train_dset = SpectrogramDataset(self.config.data_path.joinpath('train'),
                                            labels_df=train_df,
                                            transforms=self.transforms['train'])
        
        if val_df is not None:
            val_dset = SpectrogramDataset(self.config.data_path.joinpath('train'),
                                          labels_df=val_df,
                                          transforms=self.transforms['val'])
        
        if test_df is not None:
            test_dset = SpectrogramDataset(self.config.data_path.joinpath('test'),
                                           labels_df=test_df,
                                           transforms=self.transforms['val'])
        
        return {'train': train_dset, 'val': val_dset, 'test': test_dset}

    def train_dataloader(self):
       print("TRAIN DATALOADER:" + str(self.datasets['train'] is None))
       if self.datasets['train']:
        return DataLoader(dataset=self.datasets['train'],
                          batch_size=self.config.batch_size,
                          shuffle=True,
                          num_workers=self.config.num_workers)
       else:
           return None
    
    def val_dataloader(self):
        print("VAL DATALOADER:" + str(self.datasets['val'] is None))
        if self.datasets['val']:
            return DataLoader(dataset=self.datasets['val'],
                            batch_size=self.config.batch_size,
                            shuffle=False,
                            num_workers=self.config.num_workers)
        else:
            return None
    
    def predict_dataloader(self):
        print("PREDICT DATALOADER:" + str(self.datasets['test'] is None))
        if self.datasets['train']:
            return DataLoader(dataset=self.datasets['test'],
                            batch_size=self.config.batch_size,
                            shuffle=False,
                            num_workers=self.config.num_workers)
        else:
            return None
