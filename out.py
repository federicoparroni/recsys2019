import pandas as pd
import os
import time
import data

def create_sub(recommendations, submission_name, mode, directory='submissions'):
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
    handle_df.drop(cols_to_drop, axis=1, inplace=True)
    predictions_column = []
    # append the recommendations joined by spaces
    for p in recommendations:
        predictions_column.append(' '.join(map(str,p[1])))
    # add the new column to the dataframe
    handle_df['item_recommendations'] = predictions_column
    handle_df.to_csv(path_time, index=False)

    _time = time.time()-start
    elapsed = time.strftime('%Mm %Ss', time.gmtime(_time))
    print()
    print(f"submission created in submissions folder in {elapsed}")
