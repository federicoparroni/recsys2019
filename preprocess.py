import data
import utils.check_folder as cf
import os
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import scipy.sparse as sps
import numpy as np
import utils.get_action_score as gas


def urm_session_aware(train_df, test_df, time_weight, save_path):
    """
    create the URM considering the whole session of the user

    :param train_df:
    :param test_df:
    :param time_weight:
    :param save_path:
    :param save:
    :return:
    """

    global tw
    tw = time_weight

    accomodations_array = data.accomodations_ids()

    # fill missing clickout_item on the test dataframe
    test_df.fillna({'reference': -1}, inplace=True)
    train_df.fillna({'reference': -1}, inplace=True)

    # concatenate the train df and the test df mantaining only the columns of interest
    df = pd.concat([train_df, test_df])[['session_id', 'action_type', 'reference', 'impressions']]

    session_groups = df.groupby('session_id')
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

    print("dictionaries created\n")

    sessions_score = session_groups.apply(_compute_session_score).values

    # TODO: Can be optimized using data indptr and indeces
    urm = sps.csr_matrix((rows_count, cols_count), dtype=np.float)

    for i in tqdm(range(rows_count)):
        score_dict = sessions_score[i]
        for k in score_dict.keys():
            col_indx = col_of_accomodation[k]
            urm[i, col_indx] = score_dict[k]

    print("URM created\n")

    #check if the folder where to save exsist
    cf.check_folder('dataset/matrices')

    print('Saving urm matrix... ')
    sps.save_npz('{}/urm_{}.npz'.format(save_path, time_weight), urm)
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
    session_action = row['action_type']
    score = gas.get_action_score(session_action)

    if not isinstance(score, int):
      continue

    # weight the score by the time
    score *= weight_array[i]

    # get the reference to which assign the score
    try:
        reference_id = int(row['reference'])
    except ValueError:
        continue

    # was a test row in which we have to predict the clickout
    if reference_id == -1:
      continue

    #check if the reference is in the dictionary
    if reference_id not in scores.keys():
      scores[reference_id] = score
    else:
      scores[reference_id] += score

  return scores


def urm(train_df, test_df, path, clickout_score=5, impressions_score=1):
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

    accomodations_array = data.accomodations_ids()

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

    cf.check_folder(path)

    # save all
    print('Saving urm matrix... ')
    sps.save_npz('{}/urm_clickout.npz'.format(path), urm)
    print('done!')

    print('Saving row dictionary... ')
    np.save('{}/dict_row.npy'.format(path), row_of_sessionid)
    print('done!')

    print('Saving col dictionary... ')
    np.save('{}/dict_col.npy'.format(path), col_of_accomodation)
    print('done!')

def create_full_handle(test_df, name='handle.csv', folder='dataset/preprocessed/full'):
    """
    create the HANDLE CSV of the following format: |user_id,session_id,timestamp,step,impressions|

    :param test_df:
    :param local:
    :param save:
    :param name:
    :param folder:
    :return:
    """
    # user_id,session_id,timestamp,step,reference,impressions
    df_handle = test_df[['user_id', 'session_id', 'timestamp', 'step', 'impressions']]
    df_handle = df_handle[(test_df['action_type'] == 'clickout item') & (test_df['reference'].isnull())]
    print('handle created...')


    cf.check_folder(folder)
    df_handle.to_csv('{}/{}'.format(folder, name), index=False)
    print('handle saved...')


def create_small_dataset(df, maximum_rows=5000):
    """
    return a dataframe from the original dataset containing a maximum number of rows
    :param df: dataframe
    :param maximum_rows:
    
    :return: dataframe
    """
    if len(df) < maximum_rows:
      return df
    # get the last row
    row = df.iloc[maximum_rows]
    # slice the dataframe from the target row until the end
    temp_df = df.loc[maximum_rows:]
    # get the index of the last row of the last session
    end_idx = temp_df[(temp_df.session_id == row.session_id) & (temp_df.user_id == row.user_id)].index.max()
    # slice from the first row to the final index
    return df.loc[0:end_idx]


def split(df, save_path, perc_train=80):
    """
    split a dataset into train and test and create the handle of the test file
    also save the train test and the handle created
    handle as the following format |user_id,session_id,timestamp,step,clickout_item,impressions|

    :param df: dataframe to split in train and test
    :param save_path: path where to save
    :param perc_train: percentage of the df to keep in the TRAIN split
    :return:
    """
    # train-test split
    sorted_session_ids = df.groupby('session_id').first().sort_values('timestamp').reset_index()['session_id']
    slice_sorted_session_ids = sorted_session_ids.head(int(len(sorted_session_ids) * (perc_train / 100)))
    df_train = df.loc[df['session_id'].isin(slice_sorted_session_ids)]
    df_test = df.loc[~df['session_id'].isin(slice_sorted_session_ids)]

    # remove clickout from test and save an handle
    groups = df_test[df_test['action_type'] == 'clickout item'].groupby('user_id', as_index=False)
    remove_reference_tuples = groups.apply(lambda x: x.sort_values(by=['timestamp'], ascending=True).tail(1))
    df_handle = df.loc[
        [e[1] for e in remove_reference_tuples.index.tolist()], ['user_id', 'session_id', 'timestamp', 'step',
                                                                 'reference', 'impressions']]
    for e in remove_reference_tuples.index.tolist():
        df_test.at[e[1], 'reference'] = np.nan

    # save them all
    df_train.to_csv(save_path + "/train.csv", index=False)
    df_test.to_csv(save_path + "/test.csv", index=False)
    df_handle.to_csv(save_path + "/handle.csv", index=False)


def append_missing_accomodations(mode):
  found_ids = []
  
  train = data.train_df(mode)
  test = data.test_df(mode)

  for ref in train['reference'].values:
    try:
      v = int(ref)
      found_ids.append(v)
    except ValueError:
      continue
  
  for ref in test['reference'].values:
    try:
      v = int(ref)
      found_ids.append(v)
    except ValueError:
      continue

  train = train[data.train_df(mode).impressions.notnull()]
  l = [list(map(int, e.split('|'))) for e in train['impressions'].values]
  l = [item for sublist in l for item in sublist]
  found_ids.extend(l)

  test = test[data.test_df(mode).impressions.notnull()]
  l = [list(map(int, e.split('|'))) for e in test['impressions'].values]
  l = [item for sublist in l for item in sublist]
  found_ids.extend(l)

  found_ids = set(found_ids)
  acs = data.accomodations_df()
  accomod_known = set(map(int, acs['item_id'].values))
  missing = found_ids.difference(accomod_known)

  # add those at the end of the dataframe
  lst_dict = []
  for m in missing:
      lst_dict.append({'item_id':m, 'properties':np.nan})
  
  new_acs = acs.append(pd.DataFrame(lst_dict), ignore_index=True)
  new_acs.to_csv(data.ITEMS_PATH, index=False)


def preprocess():
    """
    call to create the CSV files and the URM

    NOTE: is required to have the original CSV files in the folder dataset/original
    """

    print("Hello buddy... Copenaghen is waiting...\n ")
    print()

    print("Do you want to create the CSV files ?")
    print('(1) YES, sure')
    print('(2) NO')
    choice = input()[0]

    if choice == '1':
        print('creating CSV...')

        df_train_full = data.train_df('full')
        df_small = create_small_dataset(df_train_full)

        local_path = 'dataset/preprocessed/local'
        small_path = 'dataset/preprocessed/small'
        full_path = 'dataset/preprocessed/full'

        #check if the folders exist
        cf.check_folder(local_path)
        cf.check_folder(small_path)
        cf.check_folder(full_path)

        split(df_train_full, save_path=local_path)
        split(df_small, save_path=small_path)

        #create the handle for the full test
        create_full_handle(df_train_full)

        append_missing_accomodations('full')

    elif choice == '2':
        pass
    else:
        print('Wrong choice buddy ;)')
        exit(0)

    print()
    print('(1) Create URM from LOCAL dataset')
    print('(2) Create URM from FULL dataset')
    print('(3) Create URM from SMALL dataset')
    print('(4) Don\'t create any URM')
    choice = input()[0]

    #initialize the train and test df
    train = None
    test = None
    path = None

    if choice == '1':
        path = "dataset/matrices/local"
        train = data.train_df('local')
        test = data.test_df('local')
        print('LOCAL DATASET LOADED BUDDY')
    elif choice == '2':
        path = "dataset/matrices/full"
        train = data.train_df('full')
        test = data.test_df('full')
        print('FULL DATASET LOADED BUDDY')
    elif choice == '3':
        path = "dataset/matrices/small"
        train = data.train_df('small')
        test = data.test_df('small')
        print('SMALL DATASET LOADED BUDDY')
    else:
        print('Wrong choice buddy ;)')
        exit(0)

    print()
    print('which URM do you want create buddy?')
    print()
    print('(1) Create sequence aware URM')
    print('(2) Create clickout_only URM')
    choice = input()[0]

    if choice == '1':
        """
        NOTE: CHANGE THE PARAMETERS OF THE SEQUENCE AWARE URM HERE !!!!
        """
        urm_session_aware(train, test, time_weight='lin', save_path=path)
    elif choice == '2':
        """
        NOTE: CHANGE THE PARAMETERS OF THE CLICKOUT_ONLY URM HERE !!!!
        """
        urm(train, test, path, clickout_score=5, impressions_score=1)
    else:
        print('Wrong choice buddy ;)')
        exit(0)


if __name__ == '__main__':
    """
    RUN THIS FILE TO CREATE THE CSV AND THE URM
    """
    preprocess()
