from typing import Union

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from pathlib import Path

def visualize_time_series(time_series_data: np.ndarray,
                          out_file: Union[str, Path],
                          title: str = ''):
    """
    time_series_data: (N,)
    """
    if isinstance(out_file, str):
        out_file = Path(out_file)
    
    out_file.parent.mkdir(parents=True, exist_ok=True)

    x = list(range(time_series_data.shape[0]))
    plt.plot(x, time_series_data)
    plt.title(title)

    plt.savefig(out_file)

def visualize_spectrogram(spect: np.ndarray, name: str = ''):
    """
    spect: (3, h, w)
    """
    fig, axes = plt.subplots(nrows=3, ncols=1)

    for site in range(3):
        img = plt.imshow(spect[site],
                         cmap='inferno',
                         normalize=Normalize(0, 1))
        
        axes[site] = img
        axes[site].set_title(f'{name}_site_{site}')

    plt.savefig(name)