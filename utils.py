import datetime
import pandas as pd
from pathlib import Path

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

save_train_labels_from_val()
