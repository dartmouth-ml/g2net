from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path

from spectrogram import make_spectrogram


class SpectrogramDataset(Dataset):
    # data_path is a Path object
    def __init__(self, data_path):
        super().__init__()

        labels_df = pd.read_csv(data_path.joinpath('abbreviated_labels.csv'))

        self.time_series_path = data_path.joinpath('train')
        self.file_names = labels_df['id'].tolist()
        self.labels = np.array(labels_df['target'].tolist())

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

        label = self.labels[idx]

        return spectrogram, label, file_name

    def convert_to_full_path(self, file_name):
        full_path = self.time_series_path.joinpath(*[s for s in file_name[:3]], file_name).with_suffix(self.file_ext)
        assert full_path.is_file(), full_path

        return full_path

    def __len__(self):
        return len(self.file_names)


# TODO we want separate training and validation dataloaders
def make_dataloader(batch_size):
    data_path = Path.cwd().joinpath('data', 'train')
    training_labels = pd.read_csv('./data/training_labels.csv')

    dset = SpectrogramDataset(data_path, training_labels['id'], training_labels['target'])

    return DataLoader(dataset=dset,
                      batch_size=batch_size,
                      shuffle=True)


# A little bit of testing
if __name__ == '__main__':
    dset = SpectrogramDataset(Path.cwd().joinpath('data'))
    for spectrogram, label, file_name in dset:
        print(file_name)

    spectrogram, label, file_name = dset.__getitem__(5)
    print(label)
    print(spectrogram)

    # test_series = np.load('data/train/7/7/7/777a1e4add.npy')
    # spectrogram = make_spectrogram(test_series)
