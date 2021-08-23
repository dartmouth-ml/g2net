import numpy as np
from librosa import power_to_db
from librosa.feature import melspectrogram
#import pycbc.types
from librosa.core import pseudo_cqt
from config import config


def make_spectrogram(time_series_data):
    '''Creates a MEL spectrogram.'''
    # Loop and make spectrogram
    spectrograms = []

    for i in range(3):
        if config.spect_type == 'mel':
            norm_data = time_series_data[i] / max(time_series_data[i])  # TODO is this the way we want to normalize?

            # Compute a mel-scaled spectrogram.
            spec = melspectrogram(norm_data, sr=4096, n_mels=128, fmin=20, fmax=2048)

            # Convert a power spectrogram (amplitude squared) to decibel (dB) units
            spec = power_to_db(spec).transpose((1, 0))
            spectrograms.append(spec)
        elif config.spect_type == 'pycbcq':
            #vec = time_series_data[i]
            #ts = pycbc.types.TimeSeries(vec, epoch=0, delta_t=1.0/2048) 
        
            # whiten the data (i.e. normalize the noise power at different frequencies)
            #ts = ts.whiten(0.125, 0.125)
            # calculate the qtransform
            #time, freq, power = ts.qtransform(15.0/2048, logfsteps=256, qrange=(10, 10), frange=(20, 512))
            #power -= power.min()
            #power /= power.max()
            #spectrograms.append(power)
        else:
            raise NotImplementedError(config.spect_type)


    return np.stack(spectrograms)