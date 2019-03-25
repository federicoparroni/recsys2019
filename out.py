import pandas as pd
import os
import time
import data
from tqdm import tqdm

def create_sub(predictions, submission_name, mode, directory='submissions'):
    if not os.path.exists(directory):
      os.mkdir(directory)

    tm = time.strftime('%H-%M-%S')
    path_time = f'{directory}/{submission_name} {tm}.csv'
    print(f'Exporting the sub to {path_time}...')
    start = time.time()

    handle_df = data.handle_df(mode=mode)
    # drop impressions column from the handle to mantain only the 4 keys
    cols_to_drop = ['impressions']
    if mode != 'full': # if mode is local or small, drop also the reference
        cols_to_drop.append('reference')

    # drop clickout item and impressions mantain only the 4 keys
    handle_df.drop(handle_df.columns[4], axis=1, inplace=True)
    predictions_column = []
    for p in predictions:
        predictions_column.append(str(p[1]).replace("[", '').replace("]","").replace(",", ""))
    handle_df['item_recommendations'] = predictions_column
    handle_df.to_csv(path_time, index=False)


    predictions_column = list()
    for key, value in tqdm(predictions.items()):
        predictions_column.append(str(value).replace("[", '').replace("]","").replace(",", ""))
    handle_df['item_recommendations'] = predictions_column
    handle_df.to_csv(path_time, index=False)


    time = time.time()-start
    elapsed = time.strftime('%Mm %Ss', time.gmtime(_time))
    print()
    print(f"submission created in submissions folder in {elapsed}")
