from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from librosa import power_to_db
from librosa.feature import melspectrogram
# from nnAudio.Spectrogram import CQT1992v2


def make_spectrogram(time_series_data):
    '''Creates a MEL spectrogram.'''
    # Loop and make spectrogram
    spectrograms = []

    for i in range(3):
        norm_data = time_series_data[i] / max(time_series_data[i])  # TODO is this the way we want to normalize?

        # Compute a mel-scaled spectrogram.
        spec = melspectrogram(norm_data, sr=4096, n_mels=128, fmin=20, fmax=2048)

        # Convert a power spectrogram (amplitude squared) to decibel (dB) units
        spec = power_to_db(spec).transpose((1, 0))
        spectrograms.append(spec)

    return np.stack(spectrograms)


class SpectrogramDataset(Dataset):
    def __init__(self, data_path, file_names, labels):
        super().__init__()

        self.data_path = data_path
        self.file_names = file_names
        self.labels = labels

        self.file_ext = '.npy'

    def __getitem__(self, idx):
        """
        spectrogram -- [something] x [something] image
        labels -- 0 or 1
        """
        file_name = self.file_names.iloc[idx]
        full_path = self.convert_to_full_path(file_name)
        time_series_data = np.load(full_path).astype(np.float32)
        spectrogram = make_spectrogram(time_series_data)

        label = self.labels.iloc[idx]
        label = "/".join(label.split("/")[1:])

        return spectrogram, label, file_name

    def convert_to_full_path(self, file_name):
        full_path = self.data_path.joinpath(*[s for s in file_name[:3]], file_name).with_suffix(self.file_ext)
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


if __name__ == '__main__':
    test_series = np.load('data/train/7/7/7/777a1e4add.npy')
    spectrogram = make_spectrogram(test_series)
    print(type(spectrogram))
    print(spectrogram.shape)
