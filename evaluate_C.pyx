from tqdm import tqdm
import pandas as pd
import numpy as np

cpdef float evaluate(test_df, predictions):
    """
    :param test_df: pandas dataframe composed by user_id,session_id,timestamp,step,clickout,impressions_list
    :param predictions: session_id, ordered impressions_list
    :return:
    """

    test = np.array(test_df)

    cdef float RR, MRR, position
    cdef int target_session_count, i
    RR = 0.0
    position = 0
    MRR = 0
    target_session_count = test.shape[0]

    for i in range(target_session_count):
        position = (np.where(np.array(predictions[i][1]) == test[i, 4]))[0][0]
        RR += 1/(position+1)
    MRR = RR/target_session_count

    print("MRR is: {}".format(MRR))
    return MRR




