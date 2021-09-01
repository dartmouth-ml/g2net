import pandas as pd
import numpy as np
from pathlib import Path

"""
Produces a weighted average of g2net models
    folder - pathlib Path object to folder where submissions live
    submissions - list of filenames (e.g. "my_submission.csv")
    weights - Python list or np array of weights for the submissions. Will normalize if the sum is not 1.
    output_name - the name of the resulting file (e.g. "output.csv"), which will be written in "folder"
"""
def wills_ensembler(folder, submissions, weights, output_name):
    n = len(submissions)
    if n == 0:
        print("You must include at least one submission in the ensemble")
        return

    if n != len(weights):
        print("Error: the number of weights does not equal the number of submissions")
        return

    if sum(weights) == 0:
        print("Error: at least one weight must be nonzero")
        return

    if sum(weights) != 1:
        print("Weights don't sum to 1. Normalizing...")
        weights = np.array(weights) / sum(weights)
        print(f"New weights: {weights}")

    dfs = [pd.read_csv(folder.joinpath(submissions[i])) for i in range(n)]
    weighted_target = sum([dfs[i]["target"] * weights[i] for i in range(n)])

    new_submission = dfs[0].copy()
    new_submission["target"] = weighted_target
    new_submission.to_csv(folder.joinpath(output_name), index=False)


if __name__ == '__main__':
    folder = Path.cwd().joinpath('submissions')
    submissions = ['submission_baseline.csv',
                   'submission_bigmel.csv',
                   'submission_834.csv',
                   'submission_855a.csv',
                   'submission_860.csv',
                   'submission_861.csv',
                   'submission_864.csv',
                   'submission_866.csv',
                   'submission_869.csv']
    weights = [0.01, 0.03, 0.03, 0.05, 0.08, 0.1, 0.12, 0.16, 0.42]
    # print(sum(weights))
    output_name = "custom_ensemble.csv"

    wills_ensembler(folder, submissions, weights, output_name)

