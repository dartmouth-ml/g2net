import datetime
import pandas as pd
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import sys

newpath = str(Path(sys.path[-1]).parent)
sys.path.append(newpath)
from baseline.dataloader import SpectrogramDataset

DATA_PATH = Path(__file__).resolve().parent.parent.parent.joinpath('DMLG/g2net/data_full')
DEBUG_DATA_PATH = Path(__file__).resolve().parent.parent.parent.joinpath('DMLG/g2net/data_debug')

def get_datetime_version():
    now = datetime.datetime.now()
    res = f"{now.month}{now.day}{now.year}_{now.hour}{now.minute}{now.second}"
    return res

def save_train_labels_from_val():
    train_labels_path = DATA_PATH.joinpath('training_labels.csv')
    validation_labels_path = DATA_PATH.joinpath('validation_labels.csv')
    all_labels_path = DATA_PATH.joinpath('all_labels.csv')

    all_labels = pd.read_csv(all_labels_path)
    validation_labels = pd.read_csv(validation_labels_path)
    training_labels = pd.DataFrame(columns=['id', 'target'])

    validation_ids = set()
    ids = []
    targets = []

    for _, row in validation_labels.iterrows():
        id_, target = row.loc['id'], row.loc['target']
        validation_ids.add(id_)
    
    for _, row in all_labels.iterrows():
        id_, target = row.loc['id'], row.loc['target']

        if id_ not in validation_ids:
            ids.append(id_)
            targets.append(target)

    training_labels['id'] = ids
    training_labels['target'] = targets
    print(training_labels.shape)

    training_labels.to_csv(train_labels_path)

def make_debug_data():
    root = DATA_PATH
    all_labels_path = root.joinpath('all_labels.csv')
    
    n_samples = 64

    all_labels_df = pd.read_csv(all_labels_path)
    sample_labels = all_labels_df.sample(n=n_samples)

    debug_path = root.parent.joinpath('data_debug')
    debug_path.mkdir(parents=True, exist_ok=True)

    for _, row in sample_labels.iterrows():
        prefix = '/'.join([c for c in row['id'][:3]])
        src_path = root.joinpath('train', prefix, row['id']).with_suffix('.npy')
        dst = root.parent.joinpath('data_debug', 'train', prefix, row['id']).with_suffix('.npy')
	
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_path, dst)
    
    sample_labels.to_csv(root.parent.joinpath('data_debug', 'labels.csv'), index=False, index_label=None)

def examine_dataset_outputs():
    train_data_path = DEBUG_DATA_PATH
    labels_df_path = DEBUG_DATA_PATH.joinpath('labels.csv')
    labels_df = pd.read_csv(labels_df_path)

    ds = SpectrogramDataset(
        train_data_path,
        labels_df,
        rescale=[-1, 1],
        bandpass=[20, 500], 
        return_time_series=True)

    n = 5

    for i in range(n):
        spec, ts, label, filename = ds[i]
        og_ts = np.load(filename).astype(np.float32)

        plt.subplot(1, 2, 1)
        x = list(range(og_ts.shape[1]))
        plt.plot(x, og_ts[0, ...])
        plt.title('original')

        plt.subplot(1, 2, 2)
        plt.plot(x, ts[0, ...])
        plt.title('bandpass_normalize')

        output_path = Path(__file__).parent.joinpath(f"vis/rescale_bandpass_{i}.png")
        output_path.parent.mkdir(exist_ok=True)

        plt.suptitle(f'label: {label}, {filename.name}')
        plt.savefig(output_path)
        plt.clf()


if __name__ == "__main__":
    examine_dataset_outputs()





    
