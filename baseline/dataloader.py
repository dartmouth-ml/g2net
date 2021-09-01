from typing import (
    List, 
    Union, 
    Dict
)

from functools import partial
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, sosfilt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import ToTensor, Compose
import einops

from pytorch_lightning import LightningDataModule
from common.spectrogram import get_spectogram

torch.multiprocessing.set_sharing_strategy('file_system')

class SpectrogramDataset(Dataset):
    def __init__(self,
                 data_path: Path,
                 labels_df: pd.DataFrame,
                 spec_type: str = 'mel',
                 spec_kwargs: Dict = {},
                 rescale: Union[List, None] = None,
                 bandpass: Union[List, None] = None,
                 return_time_series: bool = False,
                 transforms=None):
        super().__init__()

        self.time_series_path = data_path
        self.spec_type = spec_type
        self.spec_kwargs = spec_kwargs

        self.file_names = labels_df['id'].tolist()
        self.labels = np.array(labels_df['target'].tolist())

        self.return_time_series = return_time_series
        self.transforms = transforms
        self.file_ext = '.npy'
        
        self.preprocess_fns = []
        if rescale is not None:
            self.rescaler = MinMaxScaler(feature_range=rescale)
            self.preprocess_fns.append(partial(self.rescale_fn,
                                               rescaler=MinMaxScaler(feature_range=rescale)))
        if bandpass is not None:
            self.preprocess_fns.append(partial(self.bandpass_fn,
                                               lowcut=bandpass[0],
                                               highcut=bandpass[1],
                                               fs=4096,
                                               order=5))

    def preprocess(self, time_series_data):
        for i in range(time_series_data.shape[0]):
            for fn in self.preprocess_fns:
                time_series_data[i, ...] = fn(time_series_data[i, ...])

        return time_series_data
    
    def bandpass_fn(self, time_series_data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        sos = butter(N=order, Wn=[low, high], btype='bandpass', output='sos')
        return sosfilt(sos, time_series_data)

    def rescale_fn(self, time_series_data, rescaler):
        data = einops.rearrange(time_series_data, 'n -> n 1')
        rescaled = rescaler.fit_transform(data)
        time_series_data = einops.rearrange(rescaled, 'n 1 -> n')

        return time_series_data
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        full_path = self.convert_to_full_path(file_name)

        # load and preprocess on time domain (rescaling + bandpassing)
        time_series_data = np.load(full_path).astype(np.float32)
        time_series_data = self.preprocess(time_series_data)

        # time -> frequency
        spectrograms = get_spectogram(time_series_data,
                                      self.spec_type,
                                      window=self.window)

        label = self.labels[idx]

        # data augmentation + (np -> tensor)
        if self.transforms is not None:
            spectrograms = self.transforms(spectrograms)

        if self.return_time_series:
            return spectrograms, time_series_data, label, full_path
        else:
            return spectrograms, label, full_path

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
                                            rescale=self.config.rescale,
                                            bandpass=self.config.bandpass,
                                            do_tukey=self.config.do_tukey,
                                            transforms=self.transforms['train'])
                    
        if val_df is not None:
            val_dset = SpectrogramDataset(self.config.data_path.joinpath('train'),
                                          labels_df=val_df,
                                          rescale=self.config.rescale,
                                          bandpass=self.config.bandpass,
                                          do_tukey=self.config.do_tukey,
                                          transforms=self.transforms['val'])
        
        if test_df is not None:
            test_dset = SpectrogramDataset(self.config.data_path.joinpath('test'),
                                           labels_df=test_df,
                                           rescale=self.config.rescale,
                                           bandpass=self.config.bandpass,
                                           do_tukey=self.config.do_tukey,
                                           transforms=self.transforms['val'])
        
        return {'train': train_dset, 'val': val_dset, 'test': test_dset}

    def collate_fn(self, batch):
        batch_outputs = zip(*batch)
        if not self.config.return_time_series:
            spectograms, labels, filenames = batch_outputs
        else:
            spectograms, _, labels, filenames = batch_outputs

        spectograms_and_labels = list(zip(spectograms, labels))
        spectograms_and_labels = default_collate(spectograms_and_labels)

        spectorgrams = spectograms_and_labels[0]
        labels = spectograms_and_labels[1]

        return spectorgrams, labels, filenames

    def get_dataloader(self, mode='train'):
        shuffle = False
        if mode == 'train':
            shuffle = True
        
        dataloader_kwargs = {
            "dataset": self.datasets['mode'],
            "batch_size": self.config.batch_size,
            "shuffle": shuffle,
            "num_workers": self.config.num_workers,
            "collate_fn": self.collate_fn
        }

        return DataLoader(**dataloader_kwargs)
    
    def train_dataloader(self):
        return self.get_dataloader('train')
    
    def val_dataloader(self):
        return self.get_dataloader('val')
    
    def predict_dataloader(self):
        return self.get_dataloader('test')
