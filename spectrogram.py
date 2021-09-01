import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from librosa import power_to_db
from librosa.feature import melspectrogram
from nnAudio.Spectrogram import CQT1992v2


# The baseline spectrogram method (mel). Don't mess with this one.
# TODO test if it's faster to make all spectrograms ahead of time (avoid repeat computation)
def make_spectrogram(time_series_data):
    spectrograms = []

    for i in range(3):
        norm_data = time_series_data[i] / max(time_series_data[i])  # TODO is this the way we want to normalize?

        # Compute a mel-scaled spectrogram.
        spec = melspectrogram(norm_data, sr=4096, n_mels=128, fmin=20, fmax=2048)

        # Convert a power spectrogram (amplitude squared) to decibel (dB) units
        spec = power_to_db(spec).transpose((1, 0))
        spectrograms.append(torch.from_numpy(spec))

    return torch.stack(spectrograms)


# Generic mel spectrogram method. Takes in a variety of (optional) parameters.
#   "time_series_data" - 3 x 2048 numpy array
def spectrogram_mel(time_series_data, sr=4096, n_mels=128, fmin=20, fmax=2048):
    spectrograms = []

    for i in range(3):
        norm_data = time_series_data[i] / max(time_series_data[i])

        # Compute a mel-scaled spectrogram
        spec = melspectrogram(norm_data, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax)

        # Convert a power spectrogram (amplitude squared) to decibel (dB) units
        spec = power_to_db(spec).transpose((1, 0))
        spectrograms.append(torch.from_numpy(spec))

    return torch.stack(spectrograms)



def spectrogram_CQT(time_series_data):
    '''Transforms the np_file into spectrogram.'''
    TRANSFORM = CQT1992v2(sr=2048, fmin=20,
                          fmax=1024, hop_length=32,
                          verbose=False)

    spectrograms = []

    # Create a spectrogram for each of the 3 sites
    for i in range(3):
        norm_data = (time_series_data[i] / np.max(time_series_data[i])).astype('float32')
        norm_data = torch.from_numpy(norm_data)  # make it a tensor

        channel = TRANSFORM(norm_data).squeeze()
        spectrograms.append(channel)

    return torch.stack(spectrograms)


def vertical_image_concat(images, gap=2):
    # 1. Make sure all the images are the same size
    shapes = np.stack([img.shape for img in images])
    assert np.unique(shapes, axis=0).shape[0] == 1
    h, w, c = images[0].shape

    # 2. Concatenate
    gap_piece = np.zeros((gap, w, c)) + 1
    pieces = []
    for i in range(len(images) - 1):
        pieces.append(images[i])
        pieces.append(gap_piece)
    pieces.append(images[-1])

    return np.concatenate(pieces, axis=0)


def visualize_spectrogram(spect, name='my_plot'):
    images = []

    for site in range(3):
        # 1. get the site data and convert to numpy
        data = spect[site].detach().cpu().numpy()

        # 2. rescale to [0, 1]
        data = (data - data.min())
        data = data if data.max() == 0 else data / data.max()

        # 3. convert to image
        h, w = data.shape
        img = np.zeros((h, w, 3))
        for y in range(h):
            for x in range(w):
                color = plt.cm.inferno(data[y, x])
                for channel in range(3):
                    img[y, x, channel] = color[channel]

        plt.imshow(img)
        plt.title(f'{name}_site_{site}')
        plt.savefig(f'{name}_site_{site}')
        images.append(img)

    # Put the images together and plot
    full_img = vertical_image_concat(images)
    plt.title(f'{name}')
    plt.imshow(full_img)
    plt.savefig(f'{name}')


if __name__ == '__main__':
    inputA = np.load(Path.cwd().joinpath('data_full', '0', '0', '4', '004122364.npy'))

    spect = spectrogram_mel(inputA)

    visualize_spectrogram(spect, 'mel_)
