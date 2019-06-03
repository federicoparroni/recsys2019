import sys
import os
sys.path.append(os.getcwd())

import data
import pandas as pd
import utils.sparsedf as sparsedf
from preprocess_utils.last_clickout_indices import find as find_last_clickout

from extract_features.rnn.session_label import SessionLabel

import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from keras.utils import to_categorical
from tqdm import tqdm

# from utils.df import scale_dataframe
# from sklearn.preprocessing import MinMaxScaler


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

def one_hot_df_column(df, column_label, classes, sparse=False):
    """ Substitute a dataframe column with its one-hot encoding derived columns """
    mlb = MultiLabelBinarizer(classes=classes, sparse_output=sparse)
    res = mlb.fit_transform(df[column_label].values.reshape(-1,1))
    
    if sparse:
        df[classes] = pd.SparseDataFrame(res, columns=mlb.classes_, dtype='Int8', index=df.index)
    else:
        for i,col in enumerate(mlb.classes_):
            df[col] = pd.Series(res[:,i], dtype='int8', index=df.index)
    return df.drop(column_label, axis=1)

def add_actions_custom_encoding(df):
    """
    Custom encoding for the interactions type:
                            ck,inter,search, rating,pop,price
    'clickout item':           [1, 0, 0, 0, 0, 0],
    'interaction item info':   [0, 1, 0, 0, 0, 0],
    'interaction item rating': [0, 1, 0, 1, 0, 0],
    'interaction item image':  [0, 1, 0, 0, 0, 0],
    'interaction item deals':  [0, 1, 0, 0, 0, 1],
    'search for item':         [0, 1, 1, 0, 0, 0],
    'search for destination':  [0, 0, 1, 0, 0, 0],
    'search for poi':          [0, 0, 1, 0, 0, 0]
    Return the dataframe with the additional encoding columns:  'act_clickout','act_interaction','act_search', 
                                                                'focus_rating','focus_pop','focus_price'
    """
    action_classes = ['act_clickout', 'act_interaction', 'act_search']
    focus_classes = ['focus_rating', 'focus_pop', 'focus_price']
    encoding_classes = action_classes + focus_classes
    # maps every action type to the corresponding encoding
    mapping = {
        'clickout item':           [1, 0, 0, 0, 0, 0],
        'interaction item info':   [0, 1, 0, 0, 0, 0],
        'interaction item rating': [0, 1, 0, 1, 0, 0],
        'interaction item image':  [0, 1, 0, 0, 0, 0],
        'interaction item deals':  [0, 1, 0, 0, 0, 1],
        'search for item':         [0, 1, 1, 0, 0, 0],
        'search for destination':  [0, 0, 1, 0, 0, 0],
        'search for poi':          [0, 0, 1, 0, 0, 0],
        0:                         [0, 0, 0, 0, 0, 0]
    }

    res_matrix = np.zeros( (df.shape[0], len(encoding_classes)), dtype='int8')
    # iterate over the action_type column
    i = 0
    for a in tqdm(df['action_type']):
        res_matrix[i, :] = mapping[a]
        i += 1
    # add the resulting columns
    for j,col_name in enumerate(encoding_classes):
        df[col_name] = res_matrix[:,j]
    
    return df

def aggregate_action_type_and_sof(df):
    """ Aggregate interaction focus classes and change of sort-order-filters features in 3 columns:
        1) focus_rating |= sort_rating
        2) focus_pop |= sort_pop
        3) focus_price |= sort_price
    """
    df['focus_rating'] |= df['sort_rating']
    df['focus_pop'] |= df['sort_pop']
    df['focus_price'] |= df['sort_price']
    return df.drop(['sort_rating', 'sort_pop', 'sort_price'], axis=1)

def add_accomodations_features(df, path_to_save, logic='skip', row_indices=[]):
    """
    Add the features (one-hot) to the dataframe that match the 'reference' and save the resulting dataframe.
    It is possible to specify a list of rows to skip (logic='skip'), or to join only for some rows (logic='subset').
    Return the target columns and the one-hot columns that have been added to the dataframe
    """
    # save the references series and then set the reference to NaN to skip the join on that rows
    join_data = dict()
    join_data['backup_reference_series'] = df.reference.values.copy()
    if len(row_indices) > 0:
        if logic == 'skip':
            # set to NaN the rows to be skipped
            df.loc[row_indices, 'reference'] = np.nan
        if logic == 'subset':
            # set to NaN all rows, except for the specified rows
            backup_serie = df.loc[row_indices].reference.copy()
            df.reference = np.nan            
            df.loc[row_indices, 'reference'] = backup_serie

    # cast the reference column to Int64 removing the string values
    df.reference = pd.to_numeric(df.reference, errors='coerce') #.astype('Int64')

    # one-hot encoding of the accomodations features
    attributes_df = data.accomodations_one_hot()
    # accomodations features columns
    #features_columns = attributes_df.columns
    # with open(one_hot_accomodations_features_path, 'w') as f:
    #     pickle.dump(features_columns, f)

    #original_columns = set(df.columns)

    # add the 'no-reference' column
    #df['no_reference'] = (~df.reference.fillna('').str.isnumeric()) * 1
    
    # after_one_hot_columns = set(df.columns)
    # one_hot_columns = after_one_hot_columns.difference(original_columns)
    # one_hot_columns = list(one_hot_columns.union(set(features_columns)))
    
    def post_join(chunk_df, data):
        # reset the original references
        #chunk_df.loc[:,'reference'] = data['backup_reference_series'][data['$i1']:data['$i2']]
        return chunk_df.drop('reference', axis=1)
    
    sparsedf.left_join_in_chunks(df, attributes_df, left_on='reference', right_on=None, right_index=True,
                                post_join_fn=post_join, data=join_data, path_to_save=path_to_save)

    # print('Reloading the partial dataframe...', end=' ', flush=True)
    # df = sparsedf.read(path_to_save, sparse_cols=features_columns).set_index('orig_index')
    # reset the correct references from the backup
    #df.reference = backup_reference_series

    # return the features columns and the one-hot attributes
    #return df

def merge_reference_features(df, pad_sessions_length):
    res_df = df.copy()
    # set the non-numeric references to 0 and cast to int
    res_df.loc[res_df.reference.str.isnumeric() != True, 'reference'] = 0
    res_df = res_df.astype({'reference':'int'})
    # join
    res_df = res_df.merge(data.accomodations_one_hot(), how='left', left_on='reference', right_index=True)
    # set to 0 the features of the non-joined rows
    features_cols = data.accomodations_one_hot().columns
    col_start = list(res_df.columns).index(features_cols[0])
    col_end = list(res_df.columns).index(features_cols[-1])
    res_df.loc[:, features_cols] = res_df.loc[:, features_cols].fillna(0)
    # remove the item features for the last clickout of each session: TO-DO clickout may be not the last item
    res_df.iloc[np.arange(-1,len(res_df),pad_sessions_length)[1:], col_start:col_end] = 0
    return res_df


def add_reference_labels(df, mode, classes_prefix='ref_'):
    """ Add the reference index in the impressions list as a new column for each clickout in the dataframe.
    For the clickout interactions, a 1 is placed in the column with name {classes_prefix}{reference index in impressions}.
    For the non-clickout interactions, 0s are placed in every columns with names {classes_prefix}{0...} if 
    only_clickouts is True, else set the label for all the interactions in the session.
    NOTE: this assumes that df contains groups of padded sessions of length pad_sessions_length!
    """
    f = SessionLabel(mode=mode).read_feature()
    res_df = df.merge(f.drop(['user_id','session_id'],axis=1), how='left', left_index=True, right_index=True)
    res_df = res_df.astype({'label':'int'})

    enc = OneHotEncoder(categories=[range(25)], sparse=False)
    one_hot = enc.fit_transform(res_df['label'].values.reshape(-1, 1))
    
    # add the new columns
    for c in range(25):
        refclass = '{}{}'.format(classes_prefix, c)
        res_df[refclass] = one_hot[:,c].astype('int8')
        
    return res_df.drop('label', axis=1)

def add_reference_binary_labels(df, mode, col_name='label'):
    """ Add the reference index in the impressions list as a new column for each clickout in the dataframe.
    For the clickout interactions, a 1 is placed in the column with name {classes_prefix}{reference index in impressions}.
    For the non-clickout interactions, 0s are placed in every columns with names {classes_prefix}{0...} if 
    only_clickouts is True, else set the label for all the interactions in the session.
    NOTE: this assumes that df contains groups of padded sessions of length pad_sessions_length!
    """
    f = SessionLabel(mode=mode).read_feature()
    res_df = df.merge(f.drop(['user_id','session_id'],axis=1), how='left', left_index=True, right_index=True)
    res_df['label'] = (res_df['label'] == 0) * 1
    res_df = res_df.astype({'label':'int8'})
        
    return res_df.rename(columns={'label':col_name})


def get_last_clickout(df, index_name=None, rename_index=None):
    """ Return a dataframe with the session_id as index and the reference of the last clickout of that session. """
    tqdm.pandas()
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

    df = df[cols].groupby('session_id').progress_apply(take_last_clickout, index_name=index_name, rename_index=rename_index)
    return df.dropna().astype(int)

def add_impressions_as_new_actions(df, new_rows_starting_index=99000000, drop_cols=['impressions','prices']):
    """
    Add dummy actions before each clickout to indicate each one of the available impressions.
    Prices are incorporated inside the new rows in a new column called 'impression_price'.
    Since this is the most expensive task of the creation process, a multicore implementation would allow to split
    the work between the cpu cores (but obviuosly not working!)
    """
    df['impression_price'] = 0     #np.nan
    clickout_rows = df[df.action_type == 'clickout item']
    print('Total clickout interactions found:', clickout_rows.shape[0], flush=True)

    # build a new empty dataframe containing the original one with additional rows used to append the new interactions
    columns = list(df.columns)
    # since we don't know exactly the tot rows to append, estimate it with an upper bound, later we discard the exceeding rows 
    rows_upper_bound = clickout_rows.shape[0] * 25
    # indices are expanded with a new series (starting from 'new_rows_starting_index')
    new_indices = np.concatenate([df.index.values, np.arange(new_rows_starting_index, new_rows_starting_index + rows_upper_bound)])
    res_df = pd.DataFrame(index=new_indices, columns=columns)
    # copy the original dataframe at the begininng
    res_df.iloc[0:df.shape[0]] = df.values
    # cache the column indices to access quickly at insertion time
    show_impr_col_index = columns.index('action_type')
    steps_col_index = columns.index('step')
    reference_col_index = columns.index('reference')
    price_col_index = columns.index('impression_price')
    
    j = new_rows_starting_index         # keep tracks of the inserted rows
    for _, row in tqdm(clickout_rows.iterrows()):
        # for each clickout interaction, create a group of row to write at the end of the resulting dataframe
        impressions = list(map(int, row.impressions.split('|')))
        imprs_count = len(impressions)
        prices = list(map(int, row.prices.split('|')))
        # repeat the clickout row as many times as the impressions count and set the action_type to 'show_impression'
        rows_to_set = np.tile(row.values, (imprs_count,1))
        rows_to_set[:,show_impr_col_index] = 'show_impression'
        # set the new series of reference, prices, intermediate time steps
        rows_to_set[:,reference_col_index] = impressions
        rows_to_set[:,price_col_index] = prices
        rows_to_set[:,steps_col_index] = np.linspace(row.step-1+1/imprs_count, row.step, imprs_count+1)[0:-1]
        # write the group of rows into the right location by index
        indices = np.arange(j, j+imprs_count)
        res_df.loc[indices] = rows_to_set

        j += imprs_count

    # drop the specified columns and discard the exceeding empty rows
    res_df = res_df.drop(drop_cols, axis=1).loc[0:j-1]
    return res_df.sort_values(['user_id','session_id','timestamp','step']), j

def pad_sessions(df, max_session_length):
    """ Pad/truncate each session to have the specified length (pad by adding a number of initial rows) """
    tqdm.pandas()
    
    clickouts_indices = find_last_clickout(df)
    # this is equals to the number of sessions (sessions must have at least one clickout)
    sess_count = len(clickouts_indices)
    
    # build the resulting np matrix
    cols = ['index_column'] + list(df.columns)
    res = np.zeros((sess_count * max_session_length, len(cols)), dtype='object')
    res[:,0] = -1        # index col in first column
    
    # count how many rows we are taking for each session
    c = max_session_length
    j = sess_count-1            # point to last clickout index
    cur_userid = df.at[clickouts_indices[j], 'user_id']
    cur_sessid = df.at[clickouts_indices[j], 'session_id']
    saving_rows = False
    for idx,row in tqdm(df[::-1].iterrows()):
        #print(idx)
        user_id = df.at[idx, 'user_id']
        sess_id = df.at[idx, 'session_id']
        
        same_session_as_last_iteration = (cur_userid == user_id and cur_sessid == sess_id)
        
        # check if we are in a new session
        if not same_session_as_last_iteration:
            # new session met, pass to the next clickout index
            cur_userid = user_id
            cur_sessid = sess_id
            #print('new session began:', cur_userid, cur_sessid)
            j -= 1
            if j < 0:
                break
        
        # start taking a new session when the last clickout is found
        if idx == clickouts_indices[j]:
            # restart saving rows from the current one
            #print('idx is equal to clickout index', clickouts_indices[j])
            saving_rows = True
            c = max_session_length
        
        # iterating within the same session
        if c <= max_session_length and c > 0 and saving_rows:
            # store this interaction
            row_idx = j * max_session_length + c - 1
            #print('writing in', row_idx)
            res[row_idx, 0] = idx
            res[row_idx, 1:] = row.values
            c -= 1
        else:
            saving_rows = False
        
    return pd.DataFrame(res[:,1:], columns=df.columns, index=res[:,0])

def sessions2tensor(df, drop_cols=[], return_index=False):
    """
    Build a tensor of shape (number_of_sessions, sessions_length, features_count).
    It can return also the indices of the tensor elements.
    """
    if len(drop_cols) > 0:
        sessions_values_indices_df = df.groupby('session_id').apply(lambda g: pd.Series({'tensor': g.drop(drop_cols, axis=1).values, 'indices': g.index.values}))
    else:
        sessions_values_indices_df = df.groupby('session_id').apply(lambda g: pd.Series({'tensor': g.values, 'indices': g.index.values}))
    
    #print(sessions_values_indices_df.shape)
    if return_index:
        return np.array(sessions_values_indices_df['tensor'].to_list()), np.array(sessions_values_indices_df['indices'].to_list())
    else:
        return np.array(sessions_values_indices_df['tensor'].to_list())




# ======== POST-PROCESSING ========= #

# def load_training_dataset_for_regression(mode):
#     """ Load the one-hot dataset and return X_train, Y_train """
#     path = f'dataset/preprocessed/{cluster}/{mode}'
#     X_path = os.path.join(path, 'X_train.csv')
#     Y_path = os.path.join(path, 'Y_train.csv')

#     X_train_df = pd.read_csv(X_path, index_col=0) #sparsedf.read(X_path, sparse_cols=X_sparsecols).set_index('orig_index')
#     Y_train_df = pd.read_csv(Y_path, index_col=0) #sparsedf.read(Y_path, sparse_cols=Y_sparsecols).set_index('orig_index')

#     #X_test_df = pd.read_csv(f'dataset/preprocessed/{cluster}/{mode}/X_test.csv').set_index('orig_index')

#     # turn the timestamp into the day of year
#     X_train_df.timestamp = pd.to_datetime(X_train_df.timestamp, unit='s')
#     X_train_df['dayofyear'] = X_train_df.timestamp.dt.dayofyear

#     cols_to_drop_in_X = ['user_id','session_id','timestamp','step','platform','city','current_filters']
#     cols_to_drop_in_Y = ['session_id','user_id','timestamp','step']
    
#     # scale the dataframe
#     #X_train_df = scale_dataframe(X_train_df, ['impression_price'])
#     scaler = MinMaxScaler()
#     X_train_df.loc[:,~X_train_df.columns.isin(cols_to_drop_in_X)] = scaler.fit_transform(
#         X_train_df.drop(cols_to_drop_in_X, axis=1).astype('float64').values)

#     # get the tensors and drop currently unused columns
#     X_train = sessions2tensor(X_train_df, drop_cols=cols_to_drop_in_X)
#     print('X_train:', X_train.shape)

#     Y_train = sessions2tensor(Y_train_df, drop_cols=cols_to_drop_in_Y)
#     print('Y_train:', Y_train.shape)

#     # X_test = sessions2tensor(X_test_df, drop_cols=['user_id','session_id','step','reference','platform','city','current_filters'], return_index=False)
#     # print('X_test:', X_test.shape)

#     # X_train.impression_price = X_train.impression_price.fillna(value=0)
#     # X_test.impression_price = X_test.impression_price.fillna(value=0)
    
#     return X_train, Y_train #, X_test


# def get_session_groups_indices_df(X_df, Y_df, cols_to_group=['user_id','session_id'], indices_col_name='intrctns_indices'):
#     """
#     Return a dataframe with columns 'user_id', 'session_id', 'intrctns_indices'. This is used to retrieve the
#     interaction indices from a particular session id of a user
#     """
#     return X_df.groupby(cols_to_group).apply(lambda r: pd.Series({indices_col_name: r.index.values})).reset_index()

