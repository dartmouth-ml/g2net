import numpy as np
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