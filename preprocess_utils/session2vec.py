import sys
import os
sys.path.append(os.getcwd())

import data
import pandas as pd
import utils.sparsedf as sparsedf
from utils.df import scale_dataframe
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import utils.menu as menu
#import pickle

# ======== PRE-PROCESSING ========= #

def save_accomodations_one_hot(accomodations_df, path):
    """ Save a sparse dataframe with columns: item_id  |  feature1 | feature2 | feature3 | feature4... """
    # one-hot encoding of the accomodations features
    item_ids = accomodations_df.item_id
    accomodations_df.properties = accomodations_df.properties.str.split('|')
    accomodations_df.properties = accomodations_df.properties.fillna('')

    mlb = MultiLabelBinarizer() #(sparse_output=True)
    attributes_df = mlb.fit_transform(accomodations_df.properties)
    #attributes_df = pd.SparseDataFrame(attributes_df, columns=mlb.classes_, default_fill_value=0)
    attributes_df = pd.DataFrame(attributes_df, columns=mlb.classes_)

    # re-add item_id column to get: item_id  |  one_hot_features[...]
    attributes_df = pd.concat([item_ids, attributes_df], axis=1)

    print('Saving one-hot accomodations...', end=' ', flush=True)
    attributes_df.to_csv(path, index=False)
    # with open('features1hot', 'w') as f:
    #     pickle.dump(features_columns, f)
    print('Done!')
    return attributes_df


def df2vec(df, accomodations_df, path_to_save, row_indices_to_skip_features):
    """
    Add the features (one-hot) to the dataframe and save the resulting dataframe.
    Return the target columns and the one-hot columns that have been added to the dataframe
    """
    # save the references series and then set the reference to a negative number to skip join on that rows
    backup_reference_series = df.reference.copy()
    df.at[row_indices_to_skip_features, 'reference'] = -9999

    # one-hot encoding of the accomodations features
    attributes_df = data.accomodations_one_hot()
    # accomodations features columns
    features_columns = attributes_df.columns
    # with open(one_hot_accomodations_features_path, 'w') as f:
    #     pickle.dump(features_columns, f)

    original_columns = set(df.columns)
    # substitute a dataframe column with its one-hot encoding derived columns
    def one_hot_df_column(df, column_label):
        return pd.concat([df, pd.get_dummies(df[column_label], prefix=column_label, sparse=True)], axis=1).drop([column_label], axis=1)

    # add the one-hot encoding of the action_type feature
    print('Adding one-hot columns of action_type...')
    df = one_hot_df_column(df, 'action_type')

    # add the one-hot encoding of the device feature
    print('Adding one-hot columns of device...')
    df = one_hot_df_column(df, 'device')

    # add the 'no-reference' column
    #df['no_reference'] = (~df.reference.fillna('').str.isnumeric()) * 1
    
    after_one_hot_columns = set(df.columns)
    one_hot_columns = after_one_hot_columns.difference(original_columns)
    one_hot_columns = list(one_hot_columns.union(set(features_columns)))

    # join train and accomodations!
    print('Joining the accomodations features...')
    if os.path.isfile(path_to_save):
        os.remove(path_to_save)
    
    # join using chunks
    def pre_join_fn(chunk_df, data_dict):
        data_dict['references_serie_backup'] = chunk_df.reference.copy()
        chunk_df.loc[chunk_df.astype({'reference':str}).reference.str.isnumeric() != True, 'reference'] = -9999
        chunk_df = chunk_df.astype({'reference': np.int64}).reset_index()
        return chunk_df, data_dict

    def post_join_fn(chunk_df, data_dict):
        #chunk_df.drop('item_id', axis=1, inplace=True)
        chunk_df = chunk_df.set_index('index')
        chunk_df.reference = data_dict['references_serie_backup']
        # after the left join, the right columns are set to NaN for the non-joined rows, so we need to
        # set all the one-hot features to 0 for the non-reference rows and to re-cast to int64
        #chunk_df[features_columns] = chunk_df[features_columns].fillna(value=0).astype(np.int8)
        return chunk_df
    
    # trick to join 0s to the non-numeric reference interactions
    dummy_df_row = pd.DataFrame(np.zeros((1,len(attributes_df.columns))), columns=attributes_df.columns, index=[-9999], dtype=np.int8)
    sparsedf.left_join_in_chunks(df, attributes_df.append(dummy_df_row), left_on='reference', right_on=None, right_index=True,
                                pre_join_fn=pre_join_fn, post_join_fn=post_join_fn, path_to_save=path_to_save)

    print('Reloading full dataframe...')
    full_sparse_df = sparsedf.read(path_to_save, sparse_cols=features_columns).set_index('orig_index')
    # reset the correct references from the backup
    print('Resaving with correct references...', end=' ', flush=True)
    full_sparse_df.reference = backup_reference_series
    full_sparse_df.to_csv(path_to_save)
    print('Done!')

    # return the features columns and the one-hot attributes
    return full_sparse_df, features_columns, one_hot_columns

def get_last_clickout(df, index_name=None, rename_index=None):
    """ Return a dataframe with the session_id as index and the reference of the last clickout of that session. """
    def take_last_clickout(g, index_name, rename_index):
        clickouts = g[g.action_type == 'clickout item']
        if clickouts.shape[0] == 0:
            return None
        
        last_clickout_row = clickouts.iloc[-1]
        if index_name is None or rename_index is None:
            return pd.Series({'reference': last_clickout_row.reference})
        else:
            return pd.Series({rename_index: last_clickout_row[index_name], 'reference': last_clickout_row.reference})
    
    cols = ['session_id','action_type','reference']
    if index_name is not None:
        df = df.reset_index()
        cols.insert(0, index_name)

    tqdm.pandas()
    df = df[cols].groupby('session_id').progress_apply(take_last_clickout, index_name=index_name, rename_index=rename_index)
    return df.dropna().astype(int)

"""
def get_labels(sess2vec_df, features_columns, columns_to_keep_in_labels=['orig_index']):
    #Get the labels: features of the clickout references.
    #For the clickout interactions, the target vector is the reference fearures.
    #For the non-clickout interactions, the target vector is all zeros.
    #EDIT:
    #Extracts only the labels of clickout interactions in the sessions
    
    df_of_cols_to_keep = sess2vec_df[columns_to_keep_in_labels]

    rows_to_set_mask = sess2vec_df['action_type_clickout item'] == 0
    sess2vec_df = sess2vec_df[features_columns]
    # create a sparse matrix to store the result
    labels_sparse_matrix = sps.csr_matrix(sess2vec_df.shape, dtype=np.uint8)
    # set the labels from the dataframe target columns
    labels_sparse_matrix[np.array(rows_to_set_mask).nonzero()] = sess2vec_df[rows_to_set_mask].values
    # set to 0 the non-clickout interactions
    #labels_df.loc[ (sess2vec_df['action_type_clickout item'] == 0), : ] = 0
    
    # rebuild the dataframe
    labels_sparse_df = pd.SparseDataFrame(labels_sparse_matrix)
    labels_sparse_df[columns_to_keep_in_labels] = df_of_cols_to_keep
    return labels_sparse_df
"""

def add_impressions_columns_as_new_actions(df, new_rows_starting_index=99000000):
    df['impression_price'] = -1     #np.nan
    clickout_rows = df[df.action_type == 'clickout item']
    print('Total clickout interactions found:', clickout_rows.shape[0], flush=True)
    
    temp_df = []
    for _, row in tqdm(clickout_rows.iterrows()):
        impressions = list(map(int, row.impressions.split('|')))
        prices = list(map(int, row.prices.split('|')))
        row.action_type = 'show_impression'
  
        steps = np.linspace(row.step-1+1/len(impressions),row.step,len(impressions)+1)
        for imprsn, impr_price, step in zip(impressions, prices, steps):
            # a copy is needed otherwise row seems to be taken by reference
            r = row.copy()
            r.name = new_rows_starting_index
            
            r.reference = imprsn
            r.impression_price = impr_price
            r.step = step
            new_rows_starting_index += 1

            temp_df.append(r)

    temp_df = pd.DataFrame(temp_df, columns=clickout_rows.columns)
    df = df.append(temp_df).drop(['impressions','prices'], axis=1)
    df.index = df.index.set_names(['index'])
    
    return df.sort_values(['user_id','session_id','timestamp','step']), new_rows_starting_index

def pad_sessions(df, max_session_length):
    """ Pad/truncate each session to have the specified length (pad by adding a number of initial rows) """
    tqdm.pandas()

    def pad(g, max_length):
        # remove all interactions after the last clickout
        clickout_rows = g[g.action_type == 'clickout item']
        if clickout_rows.shape[0] > 0:
            index_of_last_clickout = clickout_rows.iloc[[-1]].index.values[0]
            g = g.loc[:index_of_last_clickout]
        
        grouplen = g.shape[0]
        if grouplen <= max_length:
            # pad with zeros
            array = np.zeros((max_length, g.shape[1]), dtype=object)
            # set index to -1, user_id and session_id to the correct ones for the padded rows
            array[:,0] = -1
            array[:,1] = g.user_id.values[0]
            array[:,2] = g.session_id.values[0]
            array[-grouplen:] = g.values[-grouplen:]
        else:
            # truncate
            array = g.values[-max_length:]
        return pd.DataFrame(array, columns=g.columns)
    
    return df.reset_index().groupby('session_id').progress_apply(pad, max_length=max_session_length).set_index('index')

def sessions2tensor(df, drop_cols=[], return_index=False):
    """
    Build a tensor of shape (number_of_sessions, sessions_length, features_count).
    It can return also the indices of the tensor elements.
    """
    if len(drop_cols) > 0:
        sessions_values_indices_df = df.groupby('session_id').apply(lambda g: pd.Series({'tensor': g.drop(drop_cols, axis=1).values, 'indices': g.index.values}))
    else:
        sessions_values_indices_df = df.groupby('session_id').apply(lambda g: pd.Series({'tensor': g.values, 'indices': g.index.values}))
    
    if return_index:
        return np.array(sessions_values_indices_df['tensor'].to_list()), np.array(sessions_values_indices_df['indices'].to_list())
    else:
        return np.array(sessions_values_indices_df['tensor'].to_list())

def create_dataset_for_regression(train_df, test_df, path):
    accomodations_df = data.accomodations_df()

    # add the impressions as new interactions
    print('Adding impressions as new actions...')
    train_df, final_new_index = add_impressions_columns_as_new_actions(train_df)
    test_df, final_new_index = add_impressions_columns_as_new_actions(test_df, final_new_index)
    print('Done!')

    # pad the sessions
    print('Padding/truncating sessions...')
    MAX_SESSION_LENGTH = 70
    train_df = pad_sessions(train_df, max_session_length=MAX_SESSION_LENGTH)
    test_df = pad_sessions(test_df, max_session_length=MAX_SESSION_LENGTH)
    print('Done!')

    train_len = train_df.shape[0]
    test_len = test_df.shape[0]

    # join train and test to one-hot correctly
    full_df = pd.concat([train_df, test_df])

    print('Getting the last clickout of each session...')
    train_clickouts_df = get_last_clickout(train_df, index_name='index', rename_index='orig_index')
    train_clickouts_indices = train_clickouts_df.orig_index.values
    train_clickouts_indices.sort()
    #full_clickouts_df = get_last_clickout(full_df, index_name='index', rename_index='orig_index')
    #full_clickouts_indices_set = set(full_clickouts_df.orig_index.values)
    print('Done!')

    print('One-hot encoding the dataset...')
    full_path = os.path.join(path, 'full_vec.csv')
    full_sparse_df, features_columns, one_hot_columns = df2vec(full_df, accomodations_df, full_path, train_clickouts_indices)
    
    print()
    print('Resplitting train and test...')

    attributes_df = data.accomodations_one_hot()

    # set to 0 the features of the last-clickout rows
    def post_join(chunk_df, data_dict):
        #chunk_df.drop('item_id', axis=1, inplace=True)
        chunk_df.drop('reference', axis=1, inplace=True)
        chunk_df = chunk_df.set_index('orig_index')
        
        # after the left join, the right columns are set to NaN for the non-joined rows, so we need to
        # set all the one-hot features to 0 for the non-reference rows and to re-cast to int64
        chunk_df[features_columns] = chunk_df[features_columns].fillna(value=0).astype(np.int8)
        return chunk_df
    
    # resplit train and test and save them
    
    train_df = full_sparse_df.head(train_len)
    print('Saving train...', end=' ', flush=True)
    train_df.to_csv( os.path.join(path, 'X_train.csv'), float_format='%.4f')
    print('Done!')
    # set the columns to be placed in the labels file, the reference is mandatory because it is used below
    Y_COLUMNS = ['user_id','session_id','timestamp','step','reference']
    train_df = train_df[Y_COLUMNS].copy()
    # get the indices and references of the last clickout for each session: session_id | orig_index, reference
    #train_clickouts_indices = list(set(train_df.index.values).intersection(full_clickouts_indices_set))
    #train_clickouts_indices.sort()
    backup_train_reference_serie = train_df.loc[train_clickouts_indices].reference.copy()
    # set the non-clickout rows to a negative number, in order to skip the join
    train_df.reference = -9999
    train_df.at[train_clickouts_indices, 'reference'] = backup_train_reference_serie
    train_df = train_df.astype({'reference':int}).reset_index()
    print('Saving train labels...')
    sparsedf.left_join_in_chunks(train_df, attributes_df, left_on='reference', right_on=None, right_index=True,
                                post_join_fn=post_join, path_to_save=os.path.join(path, 'Y_train.csv'))
    del train_df
    del train_clickouts_indices
    del backup_train_reference_serie
    
    test_df = full_sparse_df.tail(test_len)
    print('Saving test...', end=' ', flush=True)
    test_df.to_csv( os.path.join(path, 'X_test.csv'), float_format='%.4f')
    print('Done!')
    # THESE LINES BELOW SAVE THE TEST LABELS (REFERENCES OF TEST.CSV), BUT THEY ARE NONE!
    """
    test_df = test_df[['session_id','reference']].copy()
    # get the indices and references of the last clickout for each session: session_id | orig_index, reference
    test_clickouts_indices = list(set(test_df.index.values).intersection(full_clickouts_indices_set))
    test_clickouts_indices.sort()
    backup_test_reference_serie = test_df.loc[test_clickouts_indices].reference.copy()
    # set the non-clickout rows to a negative number, in order to skip the join
    test_df.reference = -9999
    test_df.at[test_clickouts_indices, 'reference'] = backup_test_reference_serie
    test_df = test_df.astype({'reference':int}).reset_index()
    print('Saving test labels...')
    sparsedf.left_join_in_chunks(test_df, attributes_df, left_on='reference', right_on=None, right_index=True,
                                post_join_fn=post_join, path_to_save=os.path.join(path, 'Y_test.csv'))
    del test_df
    del test_clickouts_indices
    del backup_test_reference_serie
    """
    
    



# ======== POST-PROCESSING ========= #

def load_dataset(mode):
    from sklearn.preprocessing import MinMaxScaler
    
    """ Load the one-hot dataset and return X_train, Y_train, X_test """
    X_train_df = pd.read_csv(f'dataset/preprocessed/cluster_recurrent/{mode}/X_train.csv').set_index('orig_index')
    Y_train_df = pd.read_csv(f'dataset/preprocessed/cluster_recurrent/{mode}/Y_train.csv').set_index('orig_index')

    #X_test_df = pd.read_csv(f'dataset/preprocessed/cluster_recurrent/{mode}/X_test.csv').set_index('orig_index')

    cols_to_drop_in_train = ['user_id','session_id','step','reference','platform','city','current_filters']
    
    # scale the dataframe
    #X_train_df = scale_dataframe(X_train_df, ['impression_price'])
    scaler = MinMaxScaler()
    X_train_df.loc[:,~X_train_df.columns.isin(cols_to_drop_in_train)] = scaler.fit_transform(X_train_df.drop(cols_to_drop_in_train, axis=1))

    # get the tensors and drop currently unused columns
    X_train = sessions2tensor(X_train_df, drop_cols=cols_to_drop_in_train)
    print('X_train:', X_train.shape)

    Y_train = sessions2tensor(Y_train_df, drop_cols=['session_id','user_id','timestamp','step'])
    print('Y_train:', Y_train.shape)

    # X_test = sessions2tensor(X_test_df, drop_cols=['user_id','session_id','step','reference','platform','city','current_filters'], return_index=False)
    # print('X_test:', X_test.shape)

    # X_train.impression_price = X_train.impression_price.fillna(value=0)
    # X_test.impression_price = X_test.impression_price.fillna(value=0)
    
    return X_train, Y_train #, X_test


def get_session_groups_indices_df(X_df, Y_df, cols_to_group=['user_id','session_id'], indices_col_name='intrctns_indices'):
    """
    Return a dataframe with columns 'user_id', 'session_id', 'intrctns_indices'. This is used to retrieve the
    interaction indices from a particular session id of a user
    """
    return X_df.groupby(cols_to_group).apply(lambda r: pd.Series({indices_col_name: r.index.values})).reset_index()



if __name__ == "__main__":
    from utils.check_folder import check_folder

    mode = menu.mode_selection()
    
    train_df = data.train_df(mode, cluster='cluster_recurrent')
    test_df = data.test_df(mode, cluster='cluster_recurrent')

    folder_path = f'dataset/preprocessed/cluster_recurrent/{mode}'
    check_folder(folder_path)

    create_dataset_for_regression(train_df, test_df, folder_path)
