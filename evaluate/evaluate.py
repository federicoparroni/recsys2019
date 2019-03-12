from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import evaluate_C
import data

TEST_PATH = "dataset/sample_test.csv"
test_df = pd.read_csv(TEST_PATH)

predictions_sample = [('62991f7c78f27', [7818446,51315,2133708,119638,86359,133581,6688860,1622273,429486,1384302,16652,52786,2448490,1017368,51416,53677,3368402,9734016,149166,18337,16660,5105382,52980,52220,569711,163,75,93,98,169,201,129,117,108,170,133,80,87,181,124,114,89,84,90,134,219,112,95,132,107]),
                      ('67c4d45f56146', [6010044,5433188,5156000,4780400,5137162,9928194,7381410,4929038,5521566,10055274,2120542,6473358,3167630,4600214,2681512,1774631,5873826,4046084,4836154,8526468,6395824,7413354,6474410,38,35,54,37,73,37,37,30,71,75,64,26,25,37,40,36,72,33,23,41,56,25,63]),
                      ('e84393cf62d13', [3132957,100226,1954167,9462680,2776177,929533,42819,42475,1346336,1001753,34468,4063872,34478,42467,4635302,9659264,2346866,1001785,34448,129053,1717401,3515296,3166108,3515434,7136232])]


def evaluate(predictions, mode):
    """
    :param mode: 'local' or 'small' say which train has been used
    :param predictions: session_id, ordered impressions_list
    :return:
    """
    assert (mode == 'local' or mode == 'small')

    handle = data.handle_df(mode)
    test = np.array(handle)

    # initialize reciprocal rank value
    RR = 0

    target_session_count = test.shape[0]

    for i in tqdm(range(target_session_count)):
        position = (np.where(np.array(predictions[i][1]) == test[i, 4]))[0][0]
        RR += 1/(position+1)

    print("MRR is: {}".format(RR/target_session_count))

    return RR/target_session_count


if __name__ == '__main__':
    evaluate_C.evaluate(test_df,predictions_sample)
    evaluate(test_df,predictions_sample)

