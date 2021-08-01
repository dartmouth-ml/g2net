import pandas as pd

def create_submission(trainer, datamodule):
    model_outs = trainer.predict(datamodule)
    submission = pd.DataFrame(columns=["id", "target"])

    for pred, data_id in model_outs:
        submission["id"].append(data_id)
        submission["target"].append(pred)
    
    submission.to_csv("submission.csv")