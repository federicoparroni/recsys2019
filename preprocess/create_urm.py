import data
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import utils.check_folder as cf
import scipy.sparse as sps
import numpy as np
import utils.get_action_score as gas
import time


tw = None

def urm_session_aware(mode, cluster='no_cluster', time_weight='lin'):
    """
    Create the URM considering the whole session of a user and giving scores based on its interactions

    :param train_df:
    :param test_df:
    :param time_weight:
    :param save_path:
    :param save:
    :return:
    """
    global tw
    tw = time_weight
    save_path = 'dataset/preprocessed/{}/{}/matrices/'.format(cluster, mode)

    accomodations_array = data.accomodations_ids()

    # load the dataframes according to the mode and cluster
    train_df = data.train_df(mode=mode, cluster=cluster)
    test_df = data.test_df(mode=mode, cluster=cluster)

    # fill missing clickout_item on the test dataframe
    test_df.fillna({'reference': -1}, inplace=True)
    train_df.fillna({'reference': -1}, inplace=True)

    # concatenate the train df and the test df mantaining only the columns of interest
    df = pd.concat([train_df, test_df])[['session_id', 'action_type', 'reference', 'impressions']]

    session_groups = df.groupby(['session_id', 'user_id'])
    session_ids = list(session_groups.groups.keys())

    rows_count = len(session_groups)
    cols_count = len(accomodations_array)

    # create dictionary (k: sessionId - v: urm row)
    row_of_sessionid = {}
    for i in range(len(session_ids)):
        row_of_sessionid[session_ids[i]] = i

    # create dictionary (k: accomodationId - v: urm col)
    col_of_accomodation = {}
    for i in range(cols_count):
        col_of_accomodation[accomodations_array[i]] = i

    print('dictionaries created\n')

    tqdm.pandas()
    sessions_score = session_groups.progress_apply(_compute_session_score).values
    print("apply function done\n")

    # create the urm using data indeces and indptr
    _data = []
    indptr = [0]
    indices = []

    values_inserted = 0
    for i in tqdm(range(rows_count)):
        score_dict = sessions_score[i]
        for k in score_dict.keys():
            indices.append(col_of_accomodation[k])
            _data.append(score_dict[k])
            values_inserted += 1
        indptr.append(values_inserted)
    _urm = sps.csr_matrix((_data, indices, indptr), shape=(rows_count, cols_count))

    print("URM created\n")

    #check if the folder where to save exsist
    cf.check_folder(save_path)

    print('Saving urm matrix... ')
    sps.save_npz('{}/urm_session_aware_{}.npz'.format(save_path, time_weight), _urm)
    print('done!')

    print('Saving row dictionary... ')
    np.save('{}/dict_row.npy'.format(save_path), row_of_sessionid)
    print('done!')

    print('Saving col dictionary... ')
    np.save('{}/dict_col.npy'.format(save_path), col_of_accomodation)
    print('done!')


def _compute_session_score(df):
  global  tw
  session_len = df.shape[0]
  #get the array of the weight based on the length
  weight_array = gas.time_weight(tw, session_len)
  scores = {}

  for i in range(session_len):

    row = df.iloc[i]

    # get the reference to which assign the score
    try:
        reference_id = int(row['reference'])
    except ValueError:
        continue

    # was a test row in which we have to predict the clickout
    if reference_id == -1:
      continue

    score = gas.get_action_score(row['action_type'])

    # weight the score by the time
    score *= weight_array[i]

    #check if the reference is in the dictionary
    if reference_id not in scores.keys():
      scores[reference_id] = score
    #else:
    #  scores[reference_id] += score

  return scores


def urm(mode, cluster, clickout_score=5, impressions_score=1):
    """
    create the URM considering only the clickout_action of every session

    :param train_df:
    :param test_df:
    :param local: operate wether using local or original dataset
    :param clickout_score: score to assign to clickout items
    :param impressions_score: score to assign to impressions accomodations, must be greater than clickout_score
    :param save:
    :return: sparse matrix (sessions, accomodations) and the association dict sessionId-urm_row
    """
    assert clickout_score > impressions_score

    save_path = 'dataset/preprocessed/{}/{}/matrices/'.format(cluster, mode)

    accomodations_array = data.accomodations_ids()

    train_df = data.train_df(mode=mode, cluster=cluster)
    test_df = data.test_df(mode=mode, cluster=cluster)

    train_df = train_df[train_df['action_type'] == 'clickout item'].fillna(-1)
    test_df = test_df[test_df['action_type'] == 'clickout item'].fillna(-1)

    df = pd.concat([train_df, test_df])[['session_id', 'reference', 'impressions']]
    session_groups = df.groupby('session_id')

    session_ids = list(session_groups.groups.keys())

    df_references = session_groups.reference.apply(lambda x: list(map(int, x))).reset_index(name='references')

    df_impressions = session_groups.impressions.apply(lambda x: list(map(int, x.values[0].split('|')))).reset_index(
        name='impressions')

    # one hot of references and impressions
    mlb = MultiLabelBinarizer(accomodations_array, sparse_output=True)

    clickout_onehot = mlb.fit_transform(df_references.references)

    impr_onehot = mlb.fit_transform(df_impressions.impressions)

    urm = (clickout_score - impressions_score) * clickout_onehot + impressions_score * impr_onehot

    # create dictionary (k: sessionId - v: urm row)
    row_of_sessionid = {}
    for i in range(len(session_ids)):
        row_of_sessionid[session_ids[i]] = i

    # create dictionary (k: accomodationId - v: urm col)
    col_of_accomodation = {}
    for i in range(len(mlb.classes)):
        col_of_accomodation[mlb.classes[i]] = i

    cf.check_folder(save_path)

    # save all
    print('Saving urm matrix... ')
    sps.save_npz('{}/urm_clickout.npz'.format(save_path), urm)
    print('done!')

    print('Saving row dictionary... ')
    np.save('{}/dict_row.npy'.format(save_path), row_of_sessionid)
    print('done!')

    print('Saving col dictionary... ')
    np.save('{}/dict_col.npy'.format(save_path), col_of_accomodation)
    print('done!')


if __name__ == '__main__':
    urm_session_aware(mode='small')
