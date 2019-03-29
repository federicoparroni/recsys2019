import pandas as pd
import os
import time
import data
from tqdm import tqdm

def create_sub(predictions, submission_name, mode, cluster='no_cluster', directory='submissions'):
    """

    :param predictions: [(session_idx_0, [acc_1, acc2, acc3, ...]),(session_idx_1, [acc_1, acc2, acc3, ...]), ...]
    :param submission_name:
    :param mode: mode of current running model.
    :param cluster: name of the cluster
    :param directory: parent path to submission
    :return: None
    """

    if not os.path.exists(directory):
      os.mkdir(directory)

    tm = time.strftime('%H-%M-%S')
    path_time = f'{directory}/{submission_name} {tm}.csv'
    print(f'Exporting the sub to {path_time}...')
    start = time.time()

    train_df = data.train_df(mode=mode, cluster=cluster)
    test_df = data.test_df(mode=mode, cluster=cluster)

    full = pd.concat(objs=[train_df, test_df])

    indices = [item[0] for item in predictions]

    targets = full.loc[indices]
    targets.drop(targets.columns[4:12], axis=1, inplace=True)

    predictions_column = []
    for p in predictions:
        predictions_column.append(str(p[1]).replace("[", '').replace("]", "").replace(",", ""))

    targets['item_recommendations'] = predictions_column
    targets.to_csv(path_time, index=False)
    _time = time.time() - start
    elapsed = time.strftime('%Mm %Ss', time.gmtime(_time))
    print()
    print(f"submission created in submissions folder in {elapsed}")

