import data
import utils.check_folder as cf
import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import scipy.sparse as sps
import numpy as np
import utils.get_action_score as gas


def urm_session_aware(train_df, test_df, time_weight, save_path, save=True):
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

    accomodations_array = data.accomodations_id()
    hnd = create_full_handle(test_df, True, save=save)

    # fill missing clickout_item on the test dataframe
    test_df.fillna({'reference': -1}, inplace=True)

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

    for i in range(rows_count):
        score_dict = sessions_score[i]
        for k in score_dict.keys():
            col_indx = col_of_accomodation[k]
            urm[i ,col_indx] = score_dict[k]

        print("URM created\n")

    if save == True:
        if not os.path.exists('dataset/matrices'):
            os.mkdir('dataset/matrices')
        sps.save_npz('dataset/matrices/urm.npz', urm)
        print("URM saved")

    np.save('dataset/matrices/dict_row.npy', row_of_sessionid)
    np.save('dataset/matrices/dict_col.npy', col_of_accomodation)
    print("dictionaries saved")
    #TODO: RETURN STATEMENT


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
    reference_id = int(row['reference'])

    # was a test row in which we have to predict the clickout
    if reference_id == -1:
      continue

    #check if the reference is in the dictionary
    if reference_id not in scores.keys():
      scores[reference_id] = score
    else:
      scores[reference_id] += score

  return scores


def urm(train_df, test_df, local, clickout_score=5, impressions_score=1, save=True):
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

    accomodations_array = data.accomodations_id()

    hnd = create_full_handle(test_df, local, save=save)

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

    if save == True:
        path = 'dataset/matrices'
        if not os.path.exists(path):
            os.mkdir(path)
        path += '/'
        if local:
            path += 'local_'

        # save all
        print('Saving urm matrix... ', end='\t')
        sps.save_npz('{}urm.npz'.format(path), urm)
        print('done!')

        print('Saving row dictionary... ', end='\t')
        np.save('{}dict_row.npy'.format(path), row_of_sessionid)
        print('done!')

        print('Saving col dictionary... ', end='\t')
        np.save('dataset/matrices/dict_col.npy', col_of_accomodation)
        print('done!')

    return urm, row_of_sessionid, col_of_accomodation, hnd


def create_full_handle(test_df, local, save=True, name='handle.csv', folder='dataset/preprocessed'):
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

    if local:
        name = 'local_{}'.format(name)
    df_handle.to_csv('{}/{}'.format(folder, name), index=False)

    return df_handle


def create_small_dataset(filename, N=1010, folder='dataset', insert_index_col=False):
    """
    create TRAIN and TEST from the original dataset with few rows
    :param filename:
    :param N:
    :param folder:
    :param insert_index_col:
    :return:
    """

    inp = '{}/original/{}.csv'.format(folder, filename)
    dest = '{}/preprocessed/{}_small.csv'.format(folder, filename)

    with open(inp, 'r') as file:
        with open(dest, 'w+') as out:
          # header
          line = file.readline()
          out.write(line)

          # data
          for i in range(N):
            line = file.readline()
            if line is not None:
              if insert_index_col:
                out.write('{},'.format(i))
              out.write(line)

              print('{}/{}'.format(i+1, N), end='\r')
            else:
              break


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
    df_train.to_csv(save_path + "train.csv", index=False)
    df_test.to_csv(save_path + "test.csv", index=False)
    df_handle.to_csv(save_path + "handle.csv", index=False)


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
        df_train_full = data.train_df('full')
        # TODO: rewrite create small dataset
        df_small = create_small_dataset()

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
        #TODO: rewrite create full handle and pass the correct parameters to the method below
        create_full_handle()

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

    if choice == '1':
        train = data.train_df('local')
        test = data.test_df('local')
        print('LOCAL DATASET LOADED BUDDY')
    elif choice == '2':
        train = data.train_df('full')
        test = data.test_df('full')
        print('FULL DATASET LOADED BUDDY')
    elif choice == '3':
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
        #TODO: REWRITE THE CREATION CODE FOR THE SESSION_AWARE URM
        pass
    elif choice == '2':
        """
        NOTE: CHANGE THE PARAMETERS OF THE CLICKOUT_ONLY URM HERE !!!!
        """
        # TODO: REWRITE THE CREATION CODE FOR THE CLICKOUT_ONLY URM
        pass
    else:
        print('Wrong choice buddy ;)')
        exit(0)


if __name__ == '__main__':
    """
    RUN THIS FILE TO CREATE THE CSV AND THE URM
    """
    preprocess()


