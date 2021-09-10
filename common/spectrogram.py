from typing import (
    Union,
    Tuple
)

import numpy as np
from functools import partial

from torch import Tensor
from scipy.signal import spectrogram
from librosa import power_to_db
from librosa.feature import melspectrogram
from nnAudio.Spectrogram import CQT1992v2

def get_spectogram(time_series_data: np.ndarray,
                   spec_type: str,
                   **kwargs) -> np.ndarray:
    if spec_type == 'mel':
        spectogram_fn = partial(spectrogram_mel, **kwargs)
    elif spec_type == 'cqt':
        spectogram_fn = partial(spectrogram_CQT, **kwargs)
    elif spec_type == 'fft':
        spectogram_fn = partial(spectogram_fft, **kwargs)
    else:
        raise NotImplementedError(spec_type)
    
    spectograms = []
    for i in range(3):
        spec = spectogram_fn(time_series_data[i, ...])
        spectograms.append(spec)
    
    return np.stack(spectograms, axis=0)

def spectogram_fft(time_series_data: np.ndarray,
                   window: Union[str, Tuple, None] = None) -> np.ndarray:

    f, t, spec = spectrogram(time_series_data, fs=2048, window=window)
    return spec

def spectrogram_mel(time_series_data: np.ndarray,
                    sr: int = 2048,
                    n_mels: int = 256,
                    fmin: int = 20,
                    fmax: int = 1024,
                    win_length: int = 256,
                    n_fft: int = 512,
                    hop_length: int = 8,
                    window: Union[str, Tuple] = '') -> np.ndarray:
    """
    Generic mel spectrogram method. Takes in a variety of (optional) parameters.

    Args:
    time_series_data: (3, sampling_rate)
    """
    # Compute a mel-scaled spectrogram
    spec = melspectrogram(
        time_series_data,
        sr=sr,
        n_mels=n_mels,
        win_length=win_length,
        n_fft=n_fft,
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length,
        window=window
    )

    # Convert a power spectrogram (amplitude squared) to decibel (dB) units
    spec = power_to_db(spec).transpose((1, 0))
    return spec

def spectrogram_CQT(time_series_data: np.ndarray,
                    sr=2048,
                    fmin=20,
                    fmax=1024,
                    hop_length=32,
                    window='hann') -> np.ndarray:
    '''Transforms the np_file into cqt spectogram.'''
    cqt_fn = CQT1992v2(
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length,
        verbose=False,
        window=window,
    )

    # Create a spectrogram for each of the 3 sites
    spec = np.squeeze(cqt_fn(Tensor(time_series_data)))
    return spec