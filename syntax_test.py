import numpy as np

arr = np.array([[1, 2, 3],
                [2, 2, 3],
                [3, 2, 3]])

uniq = np.unique(arr, axis=0)
print(uniq.shape[0])
