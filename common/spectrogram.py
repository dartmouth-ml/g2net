from typing import (
    Union,
    Tuple
)

import numpy as np
from functools import partial

from librosa import power_to_db
from librosa.feature import melspectrogram
from nnAudio.Spectrogram import CQT1992v2

from common.visualize import visualize_time_series

def get_spectogram(time_series_data: np.ndarray,
                   spec_type: str,
                   **kwargs) -> np.ndarray:
    if spec_type == 'mel':
        spectogram_fn = partial(spectrogram_mel, **kwargs)
    elif spec_type == 'cqt':
        spectogram_fn = partial(spectrogram_CQT, **kwargs)
    else:
        raise NotImplementedError(spec_type)
    
    spectograms = []
    for _ in range(3):
        spec = spectogram_fn(time_series_data)
        spectograms.append(spec)
    
    return np.stack(spectograms)

def spectrogram_mel(time_series_data: np.ndarray,
                    sr: int = 4096,
                    n_mels: int = 128,
                    fmin: int = 20,
                    fmax: int = 2048,
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
        fmin=fmin,
        fmax=fmax,
        window=window
    )

    # Convert a power spectrogram (amplitude squared) to decibel (dB) units
    spec = power_to_db(spec).transpose((1, 0))
    return spec

def spectrogram_CQT(time_series_data, sr=2048, fmin=20, fmax=1024, hop_length=32, window='') -> np.ndarray:
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
    spec = np.squeeze(cqt_fn(time_series_data))
    return spec