import pandas as pd
import os
import time

def create_sub(predictions, handle_df, directory='submissions', eval=True):
    if not os.path.exists(directory):
      os.mkdir(directory)

    path_time = directory + '/' + time.strftime('%Hh-%Mm-%Ss') + '.csv'
    start = time.time()

    # drop clickout item and impressions mantain only the 4 keys
    if eval:
        handle_df.drop(handle_df.columns[[4, 5]], axis=1, inplace=True)
    else:
        handle_df.drop(handle_df.columns[4], axis=1, inplace=True)
    predictions_column = []
    for p in predictions:
        predictions_column.append(str(p[1]).replace("[", '').replace("]","").replace(",", ""))
    handle_df['item_recommendations'] = predictions_column
    handle_df.to_csv(path_time, index=False)

    _time = time.time()-start
    print()

    print("submission created in submissions folder in {}".format(time.strftime('%Mm-%Ss', time.gmtime(_time))))




if __name__ == '__main__':
    predictions_sample = [('62991f7c78f27', [7818446,51315,2133708,119638,86359,133581,6688860,1622273,429486,1384302,16652,52786,2448490,1017368,51416,53677,3368402,9734016,149166,18337,16660,5105382,52980,52220,569711,163,75,93,98,169,201,129,117,108,170,133,80,87,181,124,114,89,84,90,134,219,112,95,132,107]),
                      ('67c4d45f56146', [6010044,5433188,5156000,4780400,5137162,9928194,7381410,4929038,5521566,10055274,2120542,6473358,3167630,4600214,2681512,1774631,5873826,4046084,4836154,8526468,6395824,7413354,6474410,38,35,54,37,73,37,37,30,71,75,64,26,25,37,40,36,72,33,23,41,56,25,63]),
                      ('e84393cf62d13', [3132957,100226,1954167,9462680,2776177,929533,42819,42475,1346336,1001753,34468,4063872,34478,42467,4635302,9659264,2346866,1001785,34448,129053,1717401,3515296,3166108,3515434,7136232])]

    TEST_PATH = "dataset/sample_test.csv"
    handle_df = pd.read_csv(TEST_PATH)
    create_sub(predictions_sample, handle_df)
