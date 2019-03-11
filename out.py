import pandas as pd
import os
import time

def create_sub(predictions, handle_df, directory='submissions'):
    if not os.path.exists(directory):
      os.mkdir(directory)

    path_time = directory + '/' + time.strftime('%Hh-%Mm-%Ss') + '.csv'
    start = time.time()

    # drop clickout item and impressions mantain only the 4 keys
    handle_df.drop(handle_df.columns[4], axis=1, inplace=True)
    predictions_column = []
    for p in predictions:
        predictions_column.append(str(p[1]).replace("[", '').replace("]","").replace(",", ""))
    handle_df['item_recommendations'] = predictions_column
    handle_df.to_csv(path_time, index=False)

    _time = time.time()-start
    print()

    print("submission created in submissions folder in {}".format(time.strftime('%Mm-%Ss', time.gmtime(_time))))
