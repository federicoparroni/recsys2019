from __future__ import print_function
import data
from utils.check_folder import check_folder
import utils.menu as menu
import os
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import scipy.sparse as sps
import numpy as np
import utils.get_action_score as gas
from time import time

def create_full_df():
    """
    Save the dataframe containing train.csv and test.csv contiguosly with reset indexes. Also save the config file
    containing the number of rows in the original train.csv (max_train_idx). This is used to know which indices
    indicates train rows (idx < max_train_idx) and test rows (idx >= max_train_idx).
    """
    train_df = data.original_train_df().reset_index(drop=True)
    len_train = len(train_df)
    train_df.to_csv(data.FULL_PATH)
    del train_df

    # save config file
    config_dict = { data.TRAIN_LEN_KEY: len_train }
    with open(data.CONFIG_FILE_PATH, 'w') as file:
        pickle.dump(config_dict, file)

    with open(data.FULL_PATH, 'a') as f:
        test_df = data.original_test_df().reset_index(drop=True)
        test_df.index += len_train
        test_df.to_csv(f, header=False)   

def urm_session_aware(train_df, test_df, time_weight, save_path):
    """
    Create the URM considering the whole session of a user and giving scores based on its interactions

    :param train_df:
    :param test_df:
    :param time_weight:
    :param save_path:
    :param save:
    :return:
    """

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

    print('dictionaries created\n')

    def _compute_session_score(df, tw):
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

            # TO-DO !!!
            # was a test row in which we have to predict the clickout
            if reference_id != -1:
                score = gas.get_action_score(row['action_type'])

                # weight the score by the time
                score *= weight_array[i]

                #check if the reference is in the dictionary
                if reference_id not in scores.keys():
                    scores[reference_id] = score
                else:
                    scores[reference_id] += score

        return scores

    tqdm.pandas()
    sessions_score = session_groups.progress_apply(_compute_session_score, tw=time_weight).values
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
    check_folder(save_path)

    print('Saving urm matrix... ')
    sps.save_npz('{}/urm_{}.npz'.format(save_path, time_weight), _urm)
    print('done!')

    print('Saving row dictionary... ')
    np.save('{}/dict_row.npy'.format(save_path), row_of_sessionid)
    print('done!')

    print('Saving col dictionary... ')
    np.save('{}/dict_col.npy'.format(save_path), col_of_accomodation)
    print('done!')

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

    check_folder(path)

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

def get_small_dataset(df, maximum_rows=5000):
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
    row = df.iloc[maximum_rows]
    # slice the dataframe from the target row until the end
    temp_df = df.iloc[maximum_rows:]
    # get the index of the last row of the last session
    end_idx = temp_df[(temp_df.session_id == row.session_id) & (temp_df.user_id == row.user_id)].index.max()
    # slice from the first row to the final index
    return df.iloc[0:end_idx]

def get_target_indices(df):
    df = df[(df['action_type'] == 'clickout item') & (df['reference'].isnull())]
    return list(df.index)

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
    print('Splitting done!')

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

def create_ICM(name='icm.npz', save_path='dataset/matrices/full/'):
    """
    it creates the ICM matrix taking as input the 'item_metadata.csv'
    the matrix is saved in COO format to accomplish easy conversion to csr and csc
    a dictionary is also saved with key = item_id and values = row of icm containing the selected item

    :param name: name of the icm matrix
    :param save_path: saving path
    :param post_processing: post-processing functions to call on the newly created ICM
    :return:
    """
    print("creating ICM...\n")
    tqdm.pandas()
    attributes_df = data.accomodations_df()

    attributes_df['properties'] = attributes_df['properties'].progress_apply(
        lambda x: x.split('|') if isinstance(x, str) else x)
    attributes_df.fillna(value='', inplace=True)
    mlb = MultiLabelBinarizer()
    one_hot_attribute = mlb.fit_transform(attributes_df['properties'].values)
    one_hot_dataframe = pd.DataFrame(one_hot_attribute, columns=mlb.classes_)

    print("ICM created succesfully!\n")
    print("creating dictionary...\n")
    dict = {}
    item_ids = attributes_df['item_id'].values
    for i in tqdm(range(len(item_ids))):
        dict[item_ids[i]] = i

    print("saving ICM...\n")
    check_folder(save_path)
    sps.save_npz(save_path + name, sps.coo_matrix(one_hot_dataframe.as_matrix()))

    print("saving dictionary")
    np.save(save_path + 'icm_dict.npy', dict)

    print("Procedure ended succesfully!")

def preprocess():
    """
    Preprocess menu

    NOTE: it is required to have the original CSV files in the folder dataset/original
    """

    def _create_csvs():
        print('creating CSV...')

        # create no_cluster/full
        # TO-DO: call the no-cluster create method

        # create item_metadata in preprocess folder
        original_item_metadata = data.accomodations_original_df()
        original_item_metadata.to_csv(data.ITEMS_PATH)

        # append missing accomodations to item metadata
        append_missing_accomodations('full')

    def _preprocess_item_metadata():
        # interactively enable preprocessing function
        pre_processing_f = [ [remove_from_stars_features, False] ]
        valid_choices = [str(i) for i in range(len(pre_processing_f))]
        inp = ''
        while inp != 'x':
            menu_title = 'Choose the preprocessing function(s) to apply to the accomodations.\nPress numbers to enable/disable the options, press X to confirm.'
            options = ['Remove \'From n stars\' attributes']
            prefixes = ['✓ ' if f[1] else '  ' for f in pre_processing_f]
            inp = menu.options(options, title=menu_title, item_prefixes=prefixes, custom_exit_label='Confirm')
            if inp in valid_choices:
                selected_idx = int(inp)
                pre_processing_f[selected_idx][1] = not pre_processing_f[selected_idx][1]
        activated_prefns = [f[0] for f in pre_processing_f if f[1]]
       
        # preprocess accomodations dataframe
        preprocess_accomodations_df(activated_prefns)
    
    def _create_URM_from_local():
        path = 'dataset/matrices/local'
        train = data.train_df('local')
        test = data.test_df('local')
        print('LOCAL DATASET LOADED BUDDY')
        return path, train, test
    def _create_URM_from_full():
        path = 'dataset/matrices/full'
        train = data.train_df('full')
        test = data.test_df('full')
        print('FULL DATASET LOADED BUDDY')
        return path, train, test
    def _create_URM_from_small():
        path = 'dataset/matrices/small'
        train = data.train_df('small')
        test = data.test_df('small')
        print('SMALL DATASET LOADED BUDDY')
        return path, train, test
    
    def _create_urm_session_aware():
        """
        NOTE: CHANGE THE PARAMETERS OF THE SEQUENCE AWARE URM HERE !!!!
        """
        urm_session_aware(train, test, time_weight='lin', save_path=path)
    def _create_urm_clickout():
        """
        NOTE: CHANGE THE PARAMETERS OF THE CLICKOUT_ONLY URM HERE !!!!
        """
        urm(train, test, path, clickout_score=5, impressions_score=1)
        
    
    print("Hello buddy... Copenaghen is waiting...")
    print()

    # create full_df.csv
    check_folder(data.FULL_PATH)
    if not os.path.isfile(data.FULL_PATH):
        print('The full dataframe (index master) is missing. Creating it...', end=' ', flush=True)
        create_full_df()
        print('Done!')
    
    # create CSV files
    menu.yesno_choice(title='Do you want to create the CSV files?', callback_yes=_create_csvs)

    # preprocess item_metadata
    menu.yesno_choice(title='Do you want to preprocess the item metadata?', callback_yes=_preprocess_item_metadata)

    # create ICM
    menu.yesno_choice(title='Do you want to create the ICM matrix files?', callback_yes=create_ICM)

    # create URM
    lbls = [ 'Create URM from LOCAL dataset', 'Create URM from FULL dataset', 'Create URM from SMALL dataset', 'Skip URM creation' ]
    callbacks = [_create_URM_from_local, _create_URM_from_full, _create_URM_from_small, lambda: 0]
    res = menu.single_choice(title='What do you want to do?', labels=lbls, callbacks=callbacks, exitable=True)
    
    if res is None:
        exit(0)

    if res != 0:
        # initialize the train and test dataframes
        path, train, test = res[0], res[1], res[2]

        callbacks = [_create_urm_session_aware, _create_urm_clickout]
        menu.single_choice(title='Which URM do you want create buddy?', labels=['Sequence-aware URM', 'Clickout URM'], callbacks=callbacks)
    
    return
    

if __name__ == '__main__':
    """
    RUN THIS FILE TO CREATE THE CSV AND THE URM
    """
    preprocess()
