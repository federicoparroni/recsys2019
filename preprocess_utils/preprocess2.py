import sys
import os
sys.path.append(os.getcwd())

import data
from utils.check_folder import check_folder
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df



def get_small_dataset(df, maximum_rows=1000000):
    """
    Return a dataframe from the original dataset containing a maximum number of rows. The actual total rows
    extracted may vary in order to avoid breaking the last session.
    :param df: dataframe
    :param maximum_rows:

    :return: dataframe
    """
    if len(df) < maximum_rows:
      return df
    # get the last row
    last_row = df.iloc[[maximum_rows]]
    last_session_id = last_row.session_id.values[0]

    # OPTIMIZATION: last_user_id = last_row.user_id.values[0]

    # slice the dataframe from the target row on
    temp_df = df.iloc[maximum_rows:]
    # get the number of remaining interactions of the last session
    # OPTIMIZATION: remaining_rows = temp_df[(temp_df.session_id == last_session_id) & (temp_df.user_id == last_user_id)].shape[0]
    remaining_rows = temp_df[temp_df.session_id == last_session_id].shape[0]
    # slice from the first row to the final index
    return df.iloc[0:maximum_rows+remaining_rows]


def append_missing_accomodations(mode):
    found_ids = []

    joined_df = data.train_df(mode).append(data.test_df(mode))

    # add references if valid
    refs = joined_df.reference
    refs = refs[refs.notnull()].values
    for r in tqdm(refs):
        try:
            v = int(r)
            found_ids.append(v)
        except ValueError:
            continue

    # add impressions
    imprs = joined_df.impressions
    imprs = imprs[imprs.notnull()].values
    for i in tqdm(imprs):
        found_ids.extend(list(map(int, i.split('|'))))

    found_ids = set(found_ids)
    acs = data.accomodations_ids()
    accomod_known = set(map(int, acs))
    missing = found_ids.difference(accomod_known)
    missing_count = len(missing)
    print('Found {} missing accomodations'.format(missing_count))

    del joined_df

    # add those at the end of the dataframe
    if missing_count > 0:
        new_acc_df = pd.DataFrame({'item_id': list(missing)}, columns=['item_id', 'properties'])

        new_acs = data.accomodations_df().append(new_acc_df, ignore_index=True)
        new_acs.to_csv(data.ITEMS_PATH, index=False)
        print('{} successfully updated'.format(data.ITEMS_PATH))


def _create_csvs():
    print('creating CSV...')

    # create no_cluster/full
    path = 'dataset/preprocessed/no_cluster'
    full = data.full_df()
    train_len = data.read_config()[data.TRAIN_LEN_KEY]

    train = full.iloc[0:train_len]
    test = full.iloc[train_len:len(full)]
    target_indices = get_target_indices(test)

    check_folder('dataset/preprocessed/no_cluster/full')
    train.to_csv(os.path.join(path, 'full/train.csv'))
    test.to_csv(os.path.join(path, 'full/test.csv'))
    np.save(os.path.join(path, 'full/train_indices'), train.index)
    np.save(os.path.join(path, 'full/test_indices'), test.index)
    np.save(os.path.join(path, 'full/target_indices'), target_indices)

    no_of_rows_in_small = int(input('How many rows do you want in small.csv? '))
    train_small = get_small_dataset(train, maximum_rows=no_of_rows_in_small)
    check_folder('dataset/preprocessed/no_cluster/small')
    split(train_small, os.path.join(path, 'small'))

    check_folder('dataset/preprocessed/no_cluster/local')
    split(train, os.path.join(path, 'local'))

    # create item_metadata in preprocess folder
    original_item_metadata = data.accomodations_original_df()
    original_item_metadata.to_csv(data.ITEMS_PATH)

    # append missing accomodations to item metadata
    append_missing_accomodations('full')


def get_target_indices(df):
    df = df[(df['action_type'] == 'clickout item') & (df['reference'].isnull())]
    return df.index.values


def split(df, save_path, perc_train=80):
    """
    Split a timestamp-ordered dataset into train and test, saving them as train.csv and test.csv in the
    specififed path. Also save the target indices file containing indices of missing clickout interactions.

    :param df: dataframe to split in train and test
    :param save_path: path where to save
    :param perc_train: percentage of the df to keep in the TRAIN split
    :return:
    """
    print('Splitting...', end=' ', flush=True)
    # train-test split
    print('sorting')
    sorted_session_ids = df.groupby('session_id').first().sort_values('timestamp').reset_index()['session_id']
    print('slicing')
    slice_sorted_session_ids = sorted_session_ids.head(int(len(sorted_session_ids) * (perc_train / 100)))
    df_train = df.loc[df['session_id'].isin(slice_sorted_session_ids)]
    df_test = df.loc[~df['session_id'].isin(slice_sorted_session_ids)]

    # remove clickout from test and save an handle
    # just those who are for real into the list of impressions
    groups = df_test[df_test['action_type'] == 'clickout item'].groupby('user_id', as_index=False)
    remove_reference_tuples = groups.apply(lambda x: x.sort_values(by=['timestamp'], ascending=True).tail(1))

    for index, row in remove_reference_tuples.iterrows():
        if int(row['reference']) not in list(map(int, row['impressions'].split('|'))):
            remove_reference_tuples.drop(index, inplace=True)

    for e in remove_reference_tuples.index.tolist():
        df_test.at[e[1], 'reference'] = np.nan

    # save them all
    df_train.to_csv(os.path.join(save_path, "train.csv"))
    df_test.to_csv(os.path.join(save_path, "test.csv"))
    np.save(os.path.join(save_path, 'target_indices'), get_target_indices(df_test))
    np.save(os.path.join(save_path, 'train_indices'), df_train.index)
    np.save(os.path.join(save_path, 'test_indices'), df_test.index)
    print('Done!')


def merge_multiple_user_sessions():
    """
    preprocess the train and test df to merge sessions of same user
    and recreate the dull df after that creat csv HAVE TO BE CALLED
    """

    def merge_sessions_test(df):

        def merge_session(df):
            temp = df.copy()
            clicks = temp[temp['reference'].isnull()].index
            if len(clicks > 1):
                temp.drop(clicks[:-1], inplace=True)
            session_id = df.at[df.index.values[-1], 'session_id']
            temp['session_id'] = session_id
            temp['step'] = np.arange(1, len(temp) + 1)
            return temp

        def drop_actions_after_click(df):
            index_last_null_click = df[df['reference'].isnull()].index
            indices_to_drop = []
            for i in tqdm(index_last_null_click):
                sess_id = df.at[i, 'session_id']
                j = i + 1
                if j >= len(df):
                    break
                c_sess_id = df.at[j, 'session_id']
                while sess_id == c_sess_id:
                    indices_to_drop.append(j)
                    j += 1
                    if j >= len(df):
                        break
                    c_sess_id = df.at[j, 'session_id']
            return indices_to_drop

        # count for each user the sessions that he has
        count_users_sessions = df[['user_id', 'session_id']].drop_duplicates().groupby('user_id').count()

        # find the users with more than 1 sessions (remember the index is the user_id)
        users_more_sessions = count_users_sessions[count_users_sessions['session_id'] > 1].index

        # drop the users with more than 1 sessions from the original dataframe they will be preprocessed
        indices_to_drop = df[df['user_id'].isin(users_more_sessions)].index
        users_one_session = df.drop(indices_to_drop)

        # take from the original df all the sessions of the users with more than one session
        multiple_sessions = df[df['user_id'].isin(users_more_sessions)].sort_values(
            ['user_id', 'timestamp']).reset_index(drop=True)

        # find the users with multiple sessions that have at least one clickout missing
        sess_with_click = multiple_sessions[multiple_sessions.reference.isnull()].user_id.unique()
        multiple_sessions = multiple_sessions[multiple_sessions['user_id'].isin(sess_with_click)].sort_values(
            ['user_id', 'timestamp']).reset_index(drop=True)

        indices_to_drop = drop_actions_after_click(multiple_sessions)
        multiple_sessions.drop(indices_to_drop, inplace=True)

        multiple_sessions = multiple_sessions.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        # find the null clickouts
        null_click = multiple_sessions[multiple_sessions.reference.isnull()]
        users_ = null_click.user_id.values

        # retrieve for each user the last timestamp in which he has a clickout nan
        timestamp = null_click.timestamp.values

        # create the dictionary user: last_timestamp_click_null
        dict_last_timestamp = {}
        for i in tqdm(range(len(users_))):
            u = users_[i]
            t = timestamp[i]
            if u in dict_last_timestamp:
                if dict_last_timestamp[u] < t:
                    dict_last_timestamp[u] = t
            else:
                dict_last_timestamp[u] = t

        index_to_drop_ = []
        for i in tqdm(multiple_sessions.index):
            user = multiple_sessions.at[i, 'user_id']
            timestamp = multiple_sessions.at[i, 'timestamp']
            if dict_last_timestamp[user] < timestamp:
                index_to_drop_.append(i)

        multiple_sessions = multiple_sessions.drop(index_to_drop_).sort_values(['user_id', 'timestamp']).reset_index(
            drop=True)

        null_click = multiple_sessions[multiple_sessions.reference.isnull()]
        users_ = null_click.user_id.values
        clicks = null_click.index

        df_list = []
        p_idx = 0
        past_user = users_[0]
        for i in tqdm(range(len(clicks))):
            idx = clicks[i]
            if users_[i] == past_user:
                df_list.append(merge_session(multiple_sessions.loc[p_idx:idx]))
            else:
                p_idx = clicks[i - 1]
                past_user = users_[i]
                df_list.append(merge_session(multiple_sessions.loc[p_idx:idx]))

        df_list.append(users_one_session)
        return pd.concat(df_list).reset_index(drop=True)

    def merge_sessions_train(df):

        def merge_session(df):
            c_session = df.at[0, 'session_id']
            c_user = df.at[0, 'user_id']
            step = 1

            session_id = [c_session]
            step_arr = [1]

            for i in tqdm(df.index[1:]):
                user = df.at[i, 'user_id']
                if c_user == user:
                    session_id.append(c_session)
                    step += 1
                    step_arr.append(step)
                else:
                    c_user = user
                    c_session = df.at[i, 'session_id']
                    step = 1
                    session_id.append(c_session)
                    step_arr.append(step)
            print(df.shape)
            print(len(session_id))
            print(len(step_arr))
            df['session_id'] = session_id
            df['step'] = step_arr
            return df

        print('finding users with more sessions')
        # count for each user the sessions that he has
        count_users_sessions = df[['user_id', 'session_id']].drop_duplicates().groupby('user_id').count()

        # find the users with more than 1 sessions (remember the index is the user_id)
        users_more_sessions = count_users_sessions[count_users_sessions['session_id'] > 1].index

        print('find users 1 sess')
        # drop the users with more than 1 sessions from the original dataframe they will be preprocessed
        indices_to_drop = df[df['user_id'].isin(users_more_sessions)].index
        users_one_session = df.drop(indices_to_drop)

        print('taking users with more sessions')
        # take from the original df all the sessions of the users with more than one session
        multiple_sessions = df[df['user_id'].isin(users_more_sessions)].sort_values(
            ['user_id', 'timestamp']).reset_index(drop=True)

        # in_place
        more_sess_df = merge_session(multiple_sessions)

        print('concatting')
        return pd.concat([more_sess_df, users_one_session])

    # retrieve the train and test df full
    print('retrieving train and test full...')
    strain = data.train_df('full')
    stest = data.test_df('full')

    # retrieve the user_id of the train and the test find the user in common between the two
    users_train = set(strain['user_id'])
    users_test = set(stest['user_id'])

    users_splitted = list(users_train & users_test)

    sessions_to_move = strain[strain['user_id'].isin(users_splitted)]

    idxs_to_drop = sessions_to_move.index

    print('creating train and test to preprocess from the old ones')
    strain_new = strain.drop(idxs_to_drop)
    stest_new = pd.concat([stest, sessions_to_move])

    del strain, stest, sessions_to_move

    print('preprocess test...')
    test_final = merge_sessions_test(stest_new)
    del stest_new

    print('preprocess train...')
    train_final = merge_sessions_train(strain_new)
    del strain_new

    print('reducing memorey usage of train and test dataframes')
    train_final = reduce_mem_usage(train_final)
    test_final = reduce_mem_usage(test_final)

    print('saving len of train')
    data.save_config(data.TRAIN_LEN_KEY, train_final.shape[0])

    print('creating new full...')
    full = pd.concat([train_final, test_final]).reset_index(drop=True)
    del train_final, test_final

    print('saving full')
    full.to_csv(data.FULL_PATH)


if __name__ == '__main__':
    merge_multiple_user_sessions()
    _create_csvs()
