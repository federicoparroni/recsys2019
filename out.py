import pandas as pd
import os
import time
import data
from tqdm import tqdm

def create_sub(predictions, submission_name, directory='submissions'):
    if not os.path.exists(directory):
      os.mkdir(directory)

    path_time = directory + '/' + submission_name + ' ' + time.strftime('%Hh-%Mm-%Ss') + '.csv'
    start = time.time()

    handle_df = data.handle_df(mode='full')

    # drop clickout item and impressions mantain only the 4 keys
    handle_df.drop(handle_df.columns[4], axis=1, inplace=True)



    predictions_column = list()
    for key, value in tqdm(predictions.items()):
        predictions_column.append(str(value).replace("[", '').replace("]","").replace(",", ""))
    handle_df['item_recommendations'] = predictions_column
    handle_df.to_csv(path_time, index=False)

    _time = time.time()-start
    print()

    print("submission created in submissions folder in {}".format(time.strftime('%Mm-%Ss', time.gmtime(_time))))
