import torch
import numpy as np

class TimeShift:
    def __init__(self, apply_shift):
        self.apply_shift = apply_shift

    def __call__(self, img):
        if self.apply_shift:
            P = img.shape[0]*torch.rand(1)
            SHIFT = P.to(torch.int32)
            return np.concatenate((img[-SHIFT:], img[:-SHIFT]), axis=0)
        return img

class SpectorShift:
    def __init__(self, apply_shift):
        self.apply_shift = apply_shift

    def __call__(self, img):
        if self.apply_shift:
            P = img.shape[1]*torch.rand(1)
            SHIFT = P.to(torch.int32)
            return np.concatenate((img[:, -SHIFT:], img[:, :-SHIFT]), axis=1)
        return img