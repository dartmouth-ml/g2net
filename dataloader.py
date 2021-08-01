from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from torchvision.transforms import ToTensor, Compose
from spectrogram import make_spectrogram
import einops
from sklearn.model_selection import train_test_split

class SpectrogramDataset(Dataset):
    # data_path is a Path object
    def __init__(self, data_path, labels_df, transforms=None):
        super().__init__()
        self.time_series_path = data_path.joinpath('train')
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

        return spectrogram, label, file_name

    def convert_to_full_path(self, file_name):
        full_path = self.time_series_path.joinpath(*[s for s in file_name[:3]], file_name).with_suffix(self.file_ext)
        assert full_path.is_file(), full_path

        return full_path

    def __len__(self):
        return len(self.file_names)


# TODO we want separate training and validation dataloaders
def make_dataloader(batch_size, val_ratio=0.2):
    # split train val
    data_path = Path(__file__).parent.joinpath('data_full')
    labels_df = pd.read_csv(data_path.joinpath('training_labels.csv'))

    labels_x = labels_df.iloc[:, 0]
    labels_y = labels_df.iloc[:, 1]
    m = labels_df.shape[0]
    filenames_train, filenames_val, labels_train, labels_val = train_test_split(labels_x, 
                                                                                labels_y,
                                                                                train_size=int(m - (m * val_ratio)),
                                                                                shuffle=True)
    
    transforms = Compose([ToTensor()])
    train_dset = SpectrogramDataset(data_path,
                                    labels_df=pd.concat([filenames_train, labels_train], axis=1),
                                    transforms=transforms)

    val_dset = SpectrogramDataset(data_path,
                                  labels_df=pd.concat([filenames_val, labels_val], axis=1),
                                  transforms=transforms)

    return {
        'train': DataLoader(dataset=train_dset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(dataset=val_dset, batch_size=batch_size, shuffle=False)
    }                

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
