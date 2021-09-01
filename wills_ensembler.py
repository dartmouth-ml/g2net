import pandas as pd
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

    if n != len(weights):
        print("Error: the number of weights does not equal the number of submissions")

    if sum(weights) == 0:
        print("Error: at least one weight must be nonzero")

    if sum(weights) != 1:
        weights = weights / sum(weights)

    dfs = [pd.read_csv(folder.joinpath(submissions[i])) for i in range(n)]
    weighted_dfs = [dfs[i]["target"] * weights[i] for i in range(n)]
    weighted_result = sum(weighted_dfs)

    new_submission = dfs[0].copy()
    new_submission["target"] = weighted_result
    new_submission.to_csv(folder.joinpath(output_name))


if __name__ == '__main__':
    folder = Path.cwd().joinpath('submissions')
    submissions = ['submission_baseline.csv', 'submission_bigmel.csv']
    weights = [1, 0]
    output_name = "output.csv"

    wills_ensembler(folder, submissions, weights, output_name)


