from __future__ import print_function
import data
from utils.check_folder import check_folder
import utils.menu as menu
import os
import pickle
from tqdm import tqdm
import pandas as pd
from preprocess_utils import create_icm, create_urm
import numpy as np

def _get_sessions_with_duplicated_steps(df):
    df_dup = df[["session_id", "user_id", "step"]]
    df_dup = df_dup[df_dup["step"] == 1]
    df_dup = df_dup.groupby(['session_id', "step"]).size() \
        .sort_values(ascending=False) \
        .reset_index(name='count')

    df_dup = df_dup[df_dup["count"] > 1]
    return list(df_dup["session_id"])

def merge_duplicates(df):
    """
        merge_duplicates : It deletes from df consecutive actions of same type within same session performed on the same
                             reference. It keeps the first occurrence of those consecutive actions and for those it saves
                             how many consecutive actions are occurred in column 'frequence'. For the remaining non-consecutive
                             actions frequence is set to 1.


        :parameter

                   user_id   session_id   timestamp   step   action_type  reference ...
                    user1	    sess1	    ts1	       0	  act_t1          x     ...
                    user1	    sess1	    ts1	       1	  act_t1          x     ...
                    user1	    sess1	    ts1	       2	  act_t1          x     ...
                    user1	    sess1	    ts2	       3	  act_t2          y     ...

                    ...         ...         ...       ...      ...           ...    ...

                    user9	    sess2	    ts3	       4	  act_t2          z     ...
                    user9	    sess2	    ts4	       5	  act_t3          w     ...
                    user9	    sess2	    ts5	       6	  act_t3          w     ...

                    ...         ...         ...       ...      ...           ...    ...

                    user4	    sess3	    ts6	       11	  act_t4          k     ...
                    user4	    sess3	    ts6	       12	  act_t5          k     ...
                    user4	    sess3	    ts6	       13	  act_t4          k     ...
                    user4	    sess3	    ts6	       14	  act_t6          k     ...

        :returns   user_id   session_id   timestamp   step   action_type  reference  ...   frequence
                    user1	    sess1	    ts1	       0	  act_t1          x      ...    3
                    user1	    sess1	    ts2	       3	  act_t2          y      ...    1

                    ...         ...         ...       ...      ...           ...     ...    ...

                    user9	    sess2	    ts3	       4	  act_t2          z      ...    2
                    user9	    sess2	    ts4	       5	  act_t3          w      ...    1

                    ...         ...         ...       ...      ...           ...     ...    ...

                    user4	    sess3	    ts6	       11	  act_t4          k      ...    1
                    user4	    sess3	    ts6	       12	  act_t5          k      ...    1
                    user4	    sess3	    ts6	       13	  act_t4          k      ...    1
                    user4	    sess3	    ts6	       14	  act_t6          k      ...    1

        :param df: DataFrame to preprocess
        :return: df: preprocessed DataFrame df with 'frequence' column
        """

    init = False
    count = 0
    duplicates_indices = list()
    count_dict = dict()

    for index, row in tqdm(df.iterrows()):
        if not init:
            init = True
        else:
            if previous_row["user_id"] == row["user_id"] and previous_row["session_id"] == row["session_id"] \
             and previous_row["action_type"] == row["action_type"] and previous_row["reference"] == row["reference"] \
             and row["action_type"] != "clickout item":
                count += 1
                duplicates_indices.append(previous_index)
            else:
                if count != 0:
                    count_dict[previous_index] = count + 1
                    count = 0

        previous_row = row
        previous_index = index

    df = df.drop(duplicates_indices)
    df.loc[list(count_dict.keys()), "frequence"] = list(count_dict.values())
    df["frequence"] = df["frequence"].fillna(1)
    df = df.astype({"frequence": int})
    return df.reset_index(drop=True)


def create_full_df():
    """
    Save the dataframe containing train.csv and test.csv contiguosly with reset indexes. Also save the config file
    containing the number of rows in the original train.csv (max_train_idx). This is used to know which indices
    indicates train rows (idx < max_train_idx) and test rows (idx >= max_train_idx).
    """
    train_df = data.original_train_df().reset_index(drop=True)
    compressed = menu.yesno_choice(title='Do you want the compressed version? (no for the original full)',
                                   callback_yes=lambda: True, callback_no=lambda: False)

    ################# TRAIN; FIXING DUPLICATED SESSION_ID <-> STEP PAIRS ##################
    sessions = _get_sessions_with_duplicated_steps(train_df)
    print("Cleaning step duplication in train.csv")
    for session in tqdm(sessions):
        mask = (train_df["session_id"] == session) & (train_df["step"] == 1)
        indices = train_df.index[mask].tolist()
        indices.sort()
        indices = indices[1:]
        # this doesn't take into account the last duplication inside the session
        # It must be tackled separatly
        for i in range(len(indices) - 1):
            mask = (train_df["session_id"] == session) & (train_df.index >= indices[i]) & (
                        train_df.index < indices[i + 1])
            train_df.loc[train_df.index[mask], "session_id"] = session + "_" + str(i)

        # handling last duplication inside the same session
        mask = (train_df["session_id"] == session) & (train_df.index >= indices[len(indices) - 1])
        train_df.loc[train_df.index[mask], "session_id"] = session + "_" + str(len(indices) - 1)

    ##################################################################################

    ################# TRAIN; MERGING DUPLICATES ########################################

    if compressed:
        train_df = merge_duplicates(train_df)
    else:
        train_df["frequence"] = 1

    ##################################################################################

    len_train = train_df.shape[0]
    train_df.to_csv(data.FULL_PATH)
    del train_df

    # save config file
    data.save_config(data.TRAIN_LEN_KEY, len_train)

    with open(data.FULL_PATH, 'a', encoding='utf-8') as f:
        test_df = data.original_test_df().reset_index(drop=True)

        ################# TEST; FIXING DUPLICATED SESSION_ID <-> STEP PAIRS ##################
        sessions = _get_sessions_with_duplicated_steps(test_df)
        print("Cleaning step duplication in test.csv")
        for session in tqdm(sessions):
            mask = (test_df["session_id"] == session) & (test_df["step"] == 1)
            indices = test_df.index[mask].tolist()
            indices.sort()
            clickout_mask = (test_df.session_id == session) & (test_df.action_type == "clickout item") \
                            & (test_df.reference.isnull())
            index_prediction = test_df.index[clickout_mask].tolist()[0]
            if (index_prediction > indices[1]):
                start_index = indices[0]
                end_index = indices[1]
                mask = (test_df["session_id"] == session) & (test_df.index >= start_index) & (test_df.index < end_index)
                test_df.loc[test_df.index[mask], "session_id"] = session + "_" + str(0)
            else:
                start_index = indices[1]
                mask = (test_df["session_id"] == session) & (test_df.index >= start_index)
                test_df.loc[test_df.index[mask], "session_id"] = session + "_" + str(0)
        ##################################################################################

        ################# TEST; MERGING DUPLICATES ########################################

        if compressed:
            test_df = merge_duplicates(test_df)
        else:
            test_df["frequence"] = 1

        ####################################################################################

        ################# TEST; DELETING UNNFORMATIVE INTERACTIONS ##########################
        
        mask = (test_df["action_type"] != "clickout item") & (test_df["reference"].isnull())
        test_df = test_df.drop(test_df[mask].index)
        test_df = test_df.reset_index(drop=True)

        ####################################################################################
        test_df.index += len_train
        test_df.to_csv(f, header=False)

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
    sorted_session_ids = df.groupby('session_id').first().sort_values('timestamp').reset_index()['session_id']
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
        new_acc_df = pd.DataFrame({ 'item_id': list(missing) }, columns=['item_id', 'properties'] )
    
        new_acs = data.accomodations_df().append(new_acc_df, ignore_index=True)
        new_acs.to_csv(data.ITEMS_PATH, index=False)
        print('{} successfully updated'.format(data.ITEMS_PATH))


def preprocess_accomodations_df(preprocessing_fns):
    """
    Preprocess and save the item metadata csv using the supplied functions. Each function will be applied
    sequentially to each row of the dataframe. The function will receive as param each dataframe row and
    should return a tuple (that will be treated as the new row columns).
    """
    assert isinstance(preprocessing_fns, list)
    
    print('Processing accomodations dataframe...')
    # load and preprocess the original item_metadata.csv
    accomodations_df = data.accomodations_original_df()

    tqdm.pandas()
    for preprfnc in preprocessing_fns:
        accomodations_df = accomodations_df.progress_apply(preprfnc, axis=1, result_type='broadcast')

    print(f'Saving preprocessed accomodations dataframe to {data.ITEMS_PATH}...', end=' ', flush=True)
    accomodations_df.to_csv(data.ITEMS_PATH, index=False)
    print('Done!')

def remove_from_stars_features(row):
    """
    Removes from the ICM the 'From n Stars' columns
    """
    propts_to_remove = ['From 2 Stars','From 3 Stars','From 4 Stars']
    if isinstance(row.properties, str):
        propts = row.properties.split('|')
        return row.item_id , '|'.join([p for p in propts if p not in propts_to_remove])
    else:
        return row

def preprocess():
    """
    Preprocess menu

    NOTE: it is required to have the original CSV files in the folder dataset/original
    """

    def _create_csvs():
        print('creating CSV...')

        # create no_cluster/full
        path = 'dataset/preprocessed/no_cluster'
        full = data.full_df()
        train_len = data.read_config()[data.TRAIN_LEN_KEY]

        train = full.loc[0:train_len]
        test = full.loc[train_len+1:len(full)]
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

    def _preprocess_item_metadata():
        # interactively enable preprocessing function
        labels = ['Remove \'From n stars\' attributes']
        pre_processing_f = [ remove_from_stars_features ]
        menu_title = 'Choose the preprocessing function(s) to apply to the accomodations.\nPress numbers to enable/disable the options, press X to confirm.'
        activated_prefns = menu.options(pre_processing_f, labels, title=menu_title, custom_exit_label='Confirm')
       
        # preprocess accomodations dataframe
        preprocess_accomodations_df(activated_prefns)

    def _create_urm_session_aware():
        """
        NOTE: CHANGE THE PARAMETERS OF THE SEQUENCE AWARE URM HERE !!!!
        """
        create_urm.urm_session_aware(mode, cluster, time_weight='lin')
    def _create_urm_clickout():
        """
        NOTE: CHANGE THE PARAMETERS OF THE CLICKOUT_ONLY URM HERE !!!!
        """
        create_urm.urm(mode, cluster, clickout_score=5, impressions_score=1)

    def _merge_sessions():
        print("Merging similar sessions (same user_id and city)")
        print("Loading full_df")
        full_df = data.full_df()
        print("Sorting, grouping, and other awesome things")
        grouped = full_df.sort_values(["user_id", "timestamp"], ascending=[True, True]).groupby(["user_id", "city"])
        new_col = np.array(["" for _ in range(len(full_df))], dtype=object)
        print("Now I'm really merging...")
        for name, g in tqdm(grouped):
            s_id = g.iloc[0]["session_id"]
            new_col[g.index.values] = s_id
        print("Writing on the df")
        full_df["unified_session_id"] = pd.Series(new_col)
        print("Saving new df to file")
        with open(data.FULL_PATH, 'w', encoding='utf-8') as f:
            full_df.to_csv(f)
        data.refresh_full_df()

    print("Hello buddy... Copenaghen is waiting...")
    print()

    # create full_df.csv
    check_folder(data.FULL_PATH)
    if os.path.isfile(data.FULL_PATH):
        menu.yesno_choice('An old full dataframe has been found. Do you want to delete it and create again?', callback_yes=create_full_df)
    else:
        print('The full dataframe (index master) is missing. Creating it...', end=' ', flush=True)
        create_full_df()
        print('Done!')


    # create CSV files
    menu.yesno_choice(title='Do you want to merge similar sessions (adding unified_session_id)?', callback_yes=_merge_sessions)

    # create CSV files
    menu.yesno_choice(title='Do you want to create the CSV files?', callback_yes=_create_csvs)

    # preprocess item_metadata
    menu.yesno_choice(title='Do you want to preprocess the item metadata?', callback_yes=_preprocess_item_metadata)

    # create ICM
    menu.yesno_choice(title='Do you want to create the ICM matrix files?', callback_yes=create_icm.create_ICM)

    # create URM
    lbls = ['Create URM from LOCAL dataset', 'Create URM from FULL dataset', 'Create URM from SMALL dataset', 'Skip URM creation' ]
    callbacks = [lambda: 'local', lambda:'full', lambda: 'small', lambda: 0]
    res = menu.single_choice(title='What do you want to do?', labels=lbls, callbacks=callbacks)
    
    if res is None:
        exit(0)

    if res != 0:
        # initialize the train and test dataframes
        mode = res

        # get the cluster
        print('for which cluster do you want to create the URM ???')
        cluster = input()
        callbacks = [_create_urm_session_aware, _create_urm_clickout]
        menu.single_choice(title='Which URM do you want create buddy?', labels=['Sequence-aware URM', 'Clickout URM'], callbacks=callbacks)
    
    return
    

if __name__ == '__main__':
    """
    RUN THIS FILE TO CREATE THE CSV AND THE URM
    """
    preprocess()
