import datetime
import pandas as pd
from pathlib import Path
import random
import shutil

def get_datetime_version():
    now = datetime.datetime.now()
    res = f"{now.month}{now.day}{now.year}_{now.hour}{now.minute}{now.second}"
    return res

def save_train_labels_from_val():
    train_labels_path = Path(__file__).resolve().parent.parent.joinpath('DMLG/g2net/data_full/training_labels.csv')
    validation_labels_path = Path(__file__).resolve().parent.parent.joinpath('DMLG/g2net/data_full/validation_labels.csv')
    all_labels_path = Path(__file__).resolve().parent.parent.joinpath('DMLG/g2net/data_full/all_labels.csv')

    all_labels = pd.read_csv(all_labels_path)
    validation_labels = pd.read_csv(validation_labels_path)
    training_labels = pd.DataFrame(columns=['id', 'target'])

    validation_ids = set()
    for i, row in validation_labels.iterrows():
        id_, target = row.loc['id'], row.loc['target']
        validation_ids.add(id_)
    
    for i, row in all_labels.iterrows():
        id_, target = row.loc['id'], row.loc['target']

        if id_ not in validation_ids:
            training_labels.append({'id': id_, 'target': target}, ignore_index=True)
    
    training_labels.to_csv(train_labels_path)

def make_debug_data():
    root = Path(__file__).resolve().parent.parent.joinpath('DMLG/g2net/data_full')
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

if __name__ == "__main__":
    make_debug_data()





    
