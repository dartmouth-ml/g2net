from typing import List, Union

from math import cos
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, sosfilt

from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import ToTensor, Compose
import einops

from pytorch_lightning import LightningDataModule
from baseline.spectrogram import make_spectrogram

class SpectrogramDataset(Dataset):
    def __init__(self,
                 data_path: Path,
                 labels_df: pd.DataFrame,
                 rescale: Union[List[float], None] = None,
                 bandpass: Union[List[float], None] = None,
                 do_tukey=True,
                 return_time_series: bool = False,
                 transforms=None):
        super().__init__()
        self.time_series_path = data_path

        self.file_names = labels_df['id'].tolist()
        self.labels = np.array(labels_df['target'].tolist())

        self.return_time_series = return_time_series
        self.transforms = transforms
        self.rescale = rescale
        self.file_ext = '.npy'

        if self.rescale is not None:
            self.rescaler = MinMaxScaler(feature_range=rescale)
        
        self.do_tukey = do_tukey
        self.bandpass = bandpass

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(N=order, Wn=[low, high], btype='bandpass', output='sos')
        return sosfilt(sos, data)
    
    def tukey_fn(self, size, alpha, dtype):
        window = np.zeros((size,), dtype=dtype)
        for n in range(size // 2 + 1):
            if n < alpha * size / 2:
                window[n] = window[-n] = 1 / 2 * (1 - cos(2 * 3.14 * n / (size * alpha)))
            else:
                window[n] = window[-n] = 1
        return window

    def __getitem__(self, idx):
        """
        spectrogram -- (currently) 9 x 128 color image
        label -- 0 or 1
        """
        file_name = self.file_names[idx]
        full_path = self.convert_to_full_path(file_name)
        time_series_data = np.load(full_path).astype(np.float32)

        # rescale
        for i in range(time_series_data.shape[0]):
            if self.rescale:
                data = einops.rearrange(time_series_data[i, ...], 'n -> n 1')
                rescaled = self.rescaler.fit_transform(data)
                time_series_data[i, ...] = einops.rearrange(rescaled, 'n 1 -> n')
            if self.do_tukey:
                tukey = self.tukey_fn(time_series_data.shape[-1], 0.2, time_series_data.dtype)
                time_series_data[i, ...] *= tukey
            if self.bandpass:
                time_series_data[i, ...] = self.butter_bandpass_filter(time_series_data[i, ...],
                                                                        self.bandpass[0],
                                                                        self.bandpass[1],
                                                                        4096)
        spectrograms = make_spectrogram(time_series_data)
        spectrograms = np.stack(spectrograms, axis=0) # 3, n_mels, t

        label = self.labels[idx]
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

    def train_dataloader(self):
        return DataLoader(dataset=self.datasets['train'],
                          batch_size=self.config.batch_size,
                          shuffle=True,
                          num_workers=self.config.num_workers,
                          collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.datasets['val'],
                         batch_size=self.config.batch_size,
                         shuffle=False,
                         num_workers=self.config.num_workers,
                         collate_fn=self.collate_fn)
    
    def predict_dataloader(self):
        return DataLoader(dataset=self.datasets['test'],
                          batch_size=self.config.batch_size,
                          shuffle=False,
                          num_workers=self.config.num_workers,
                          collate_fn=self.collate_fn)
