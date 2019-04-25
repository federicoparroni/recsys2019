import os
import sys
sys.path.append(os.getcwd())

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


class urm_creator:

    def __init__(self, type, mode, cluster, name):

        assert type in ['user', 'session']
        assert mode in ['small', 'local', 'full']

        self.save_path = f'dataset/preprocessed/{cluster}/{mode}/matrices/{type}'
        cf.check_folder(self.save_path)

        self.score_dict= {
            'clickout item': 3,
            'interaction item rating': 3,
            'interaction item info': 1,
            'interaction item image': 3,
            'interaction item deals': 1,
            'search for item': 5,
            'search for destination': 'reset',
            'change of sort order': None,
            'filter selection': None,
            'search for poi': None,
            'tw': 'lin',
            'score_update_rule': 'substitute'
        }

        self.name = name
        self.type = type
        self.mode = mode
        self.cluster = cluster

        self.accomodations_id = data.accomodations_ids()
        self.train_df = None
        self.test_df = None

    def create_urm(self):

        # load the dataframes according to the mode and cluster
        train_df = data.train_df(mode=self.mode, cluster=self.cluster)
        test_df = data.test_df(mode=self.mode, cluster=self.cluster)

        # fill missing clickout_item on the test dataframe
        test_df.fillna({'reference': -1}, inplace=True)
        train_df.fillna({'reference': -1}, inplace=True)

        # concatenate the train df and the test df mantaining only the columns of interest
        df = pd.concat([train_df, test_df])[['session_id', 'user_id', 'action_type', 'reference', 'impressions']]

        if self.type == 'user':
            session_groups = df.groupby(['user_id'])
        if self.type == 'session':
            session_groups = df.groupby(['user_id', 'session_id'])

        # it coincides with the rows number
        groups_keys = list(session_groups.groups.keys())

        rows_count = len(groups_keys)
        cols_count = len(self.accomodations_id)

        """
        create ROW dictionary
            if type == USER :
                key: user_id -- value: row_urm
            if type == SESSION :
                key: (user_id, session_id) -- value: row_urm
        """
        row_dict = {}
        for i in range(rows_count):
            row_dict[groups_keys[i]] = i

        """
        create COL dictionary
            key: accomodation_id -- value: col_urm
        """
        col_dict = {}
        for i in range(cols_count):
            col_dict[self.accomodations_id[i]] = i

        print('dictionaries created\n')

        tqdm.pandas()
        # compute the score
        sessions_score = session_groups.progress_apply(self._compute_session_score).values

        print("apply function done\n")

        # create the urm using data indeces and indptr
        _data = []
        indptr = [0]
        indices = []

        values_inserted = 0
        for i in tqdm(range(rows_count)):
            score_dict = sessions_score[i]
            for k in score_dict.keys():
                indices.append(col_dict[k])
                _data.append(score_dict[k])
                values_inserted += 1
            indptr.append(values_inserted)
        _urm = sps.csr_matrix((_data, indices, indptr), shape=(rows_count, cols_count))

        print("URM created\n")

        print('Saving urm matrix... ')
        sps.save_npz(f'{self.save_path}/{self.name}.npz', _urm)
        print('done!')

        print('Saving row dictionary... ')
        np.save(f'{self.save_path}/{self.name}_dict_row.npy', row_dict)
        print('done!')

        print('Saving col dictionary... ')
        np.save(f'{self.save_path}/{self.name}_dict_col.npy', col_dict)
        print('done!')

    def _create_weight_array(self, weight_function, session_length):
        """
        :param weight_function:
        :param session_lenght:
        :return:
        """
        assert weight_function in ['exp', 'lin', None]

        weight_array = []
        if weight_function == 'exp':
            for i in range(session_length):
                weight_array.append(((i + 1) / session_length) ** 3)
            return weight_array
        if weight_function == 'lin':
            for i in range(session_length):
                weight_array.append((i + 1) / session_length)
            return weight_array
        if weight_function == None:
            for i in range(session_length):
                weight_array.append(1)
            return weight_array

    def _accomodation_score_update(self, old_value, new_value):
        r = self.score_dict['score_update_rule']
        assert r in ['sum', 'substitute', 'no_update']
        if r == 'sum':
            return old_value+new_value
        if r == 'substitute':
            return new_value
        if r == 'no_update':
            return old_value

    def _compute_session_score(self, df):
        session_len = df.shape[0]
        # get the array of the weight based on the length
        weight_array = self._create_weight_array(weight_function=self.score_dict['tw'], session_length=session_len)

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

            score = self.score_dict[row['action_type']]

            # weight the score by the time
            score *= weight_array[i]

            # check if the reference is in the dictionary
            if reference_id not in scores:
                scores[reference_id] = score
            else:
                scores[reference_id] = self._accomodation_score_update(old_value=scores[reference_id], new_value=score)

        return scores







def urm_session_aware(mode, action_score_dict, cluster='no_cluster', time_weight='lin'):
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
    df = pd.concat([train_df, test_df])[['session_id', 'user_id', 'action_type', 'reference', 'impressions']]

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
    sps.save_npz('{}/urm_session_aware1_{}.npz'.format(save_path, time_weight), _urm)
    print('done!')

    print('Saving row dictionary... ')
    np.save('{}/dict_row.npy'.format(save_path), row_of_sessionid)
    print('done!')

    print('Saving col dictionary... ')
    np.save('{}/dict_col.npy'.format(save_path), col_of_accomodation)
    print('done!')


def urm_score(type, mode, _last_click_score=10, _clicked_ref_score=5, _impr_not_seen_score=1, _seen_ref_score=-1, cluster='no_cluster'):

    """
    Create the URM considering the whole session of a user and giving scores based on its interactions

    :param train_df:
    :param test_df:
    :param time_weight:
    :param save_path:
    :param save:
    :return:
    """
    global impr_not_seen_score, last_click_score, seen_ref_score, clicked_ref_score
    impr_not_seen_score = _impr_not_seen_score
    last_click_score = _last_click_score
    clicked_ref_score = _clicked_ref_score
    seen_ref_score = _seen_ref_score


    save_path = 'dataset/preprocessed/{}/{}/matrices/'.format(cluster, mode)

    accomodations_array = data.accomodations_ids()

    # load the dataframes according to the mode and cluster
    train_df = data.train_df(mode=mode, cluster=cluster)
    test_df = data.test_df(mode=mode, cluster=cluster)

    # fill missing clickout_item on the test dataframe
    test_df.fillna({'reference': -1}, inplace=True)
    train_df.fillna({'reference': -1}, inplace=True)

    # concatenate the train df and the test df mantaining only the columns of interest
    df = pd.concat([train_df, test_df])[['session_id', 'user_id', 'action_type', 'reference', 'impressions']]

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
    sessions_score = session_groups.progress_apply(_session_score_negative_value_seen_elem).values
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
    sps.save_npz('{}/urm_negative.npz'.format(save_path), _urm)
    print('done!')

    print('Saving row dictionary... ')
    np.save('{}/dict_row.npy'.format(save_path), row_of_sessionid)
    print('done!')

    print('Saving col dictionary... ')
    np.save('{}/dict_col.npy'.format(save_path), col_of_accomodation)
    print('done!')


def urm_neg_score_user(mode, _last_click_score=1, _clicked_ref_score=1, _impr_not_seen_score=0, _seen_ref_score=1, cluster='no_cluster'):
    global impr_not_seen_score, last_click_score, seen_ref_score, clicked_ref_score
    impr_not_seen_score = _impr_not_seen_score
    last_click_score = _last_click_score
    clicked_ref_score = _clicked_ref_score
    seen_ref_score = _seen_ref_score


    save_path = 'dataset/preprocessed/{}/{}/matrices/'.format(cluster, mode)

    accomodations_array = data.accomodations_ids()

    # load the dataframes according to the mode and cluster
    train_df = data.train_df(mode=mode, cluster=cluster)
    test_df = data.test_df(mode=mode, cluster=cluster)

    # fill missing clickout_item on the test dataframe
    test_df.fillna({'reference': -1}, inplace=True)
    train_df.fillna({'reference': -1}, inplace=True)

    # concatenate the train df and the test df mantaining only the columns of interest
    df = pd.concat([train_df, test_df])[['session_id', 'user_id', 'action_type', 'reference', 'impressions']]

    session_groups = df.groupby(['user_id'])
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
    sessions_score = session_groups.progress_apply(_session_score_negative_value_seen_elem).values
    print("apply function done\n")

    # create the urm using data indeces and indptr
    _data = []
    indptr = [0]
    indices = []

    values_inserted = 0
    for i in tqdm(range(rows_count)):
        score_dict = sessions_score[i]
        for k in score_dict.keys():
            # TODO: FIND WHY THERE IS A KEY EQUAL -1
            if k != -1:
                indices.append(col_of_accomodation[k])
                _data.append(score_dict[k])
                values_inserted += 1
        indptr.append(values_inserted)
    _urm = sps.csr_matrix((_data, indices, indptr), shape=(rows_count, cols_count))

    print("URM created\n")

    #check if the folder where to save exsist
    cf.check_folder(save_path)

    print('Saving urm matrix... ')
    sps.save_npz('{}/urm_negative_user.npz'.format(save_path), _urm)
    print('done!')

    print('Saving row dictionary... ')
    np.save('{}/dict_row_user.npy'.format(save_path), row_of_sessionid)
    print('done!')

    print('Saving col dictionary... ')
    np.save('{}/dict_col_user.npy'.format(save_path), col_of_accomodation)
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
    if reference_id not in scores:
      scores[reference_id] = score
    else:
      scores[reference_id] = (scores[reference_id]+score)/2

  return scores


def _session_score_negative_value_seen_elem(df):
    # initialize the dict that will contain the scores of the references
    score = {}
    ref_last_clickout = None
    clickout_items_not_final = None
    session_impressions = None
    numeric_references = []

    # get all the impressions of the clickouts
    clickouts = df[df['action_type'] == 'clickout item']

    # check if there is at least one clickout
    if len(clickouts) > 0:
        ref_last_click = int(clickouts.tail(1)['reference'])
        if ref_last_click != -1:
            ref_last_clickout = ref_last_click

        # check if in the session there are more than 1 clickout
        if len(clickouts) > 1:
            # remove the row of last clickout
            clicked_ref = list(map(int, (clickouts['reference'].unique())))
            # remove the last click from the clicked one
            if ref_last_clickout is not None:
                clicked_ref.remove(ref_last_clickout)
            clickout_items_not_final = clicked_ref

        session_impressions = list(map(int, clickouts[clickouts['reference'] != -1]['reference'].unique()))
        # remove from the impressions the last clicked
        if ref_last_clickout is not None:
            session_impressions.remove(ref_last_clickout)

    # take all the numeric reference in the session
    references = list(df['reference'].unique())
    for r in references:
        try:
            r = int(r)
            if r != -1:
                numeric_references.append(r)
        except ValueError:
            continue

    global impr_not_seen_score, last_click_score, seen_ref_score, clicked_ref_score

    # if there isn't a clickout in the session we assume all reference are not pleased the user
    if ref_last_clickout is None:
        for ref in numeric_references:
            score[ref] = impr_not_seen_score

    # if there is 1 clickout
    if ref_last_clickout is not None:

        score[ref_last_clickout] = last_click_score
        for ref in numeric_references:
            score[ref] = seen_ref_score
        for impr in session_impressions:
            if impr not in score:
                score[impr] = impr_not_seen_score
                # if there are more than 1 clickout
        if clickout_items_not_final is not None:
            for click in clickout_items_not_final:
                score[click] = clicked_ref_score
    return score


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
    import utils.menu as menu

    mode = menu.mode_selection()

    urm_creator = urm_creator(type='session', mode=mode, cluster='no_cluster', name='urm_recurrent_models')
    urm_creator.create_urm()
