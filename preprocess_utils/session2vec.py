import sys
import os
sys.path.append(os.getcwd())

import data
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from tqdm import tqdm

# ======== PRE-PROCESSING ========= #

def get_accomodations_one_hot(accomodations_df, sparse):
    """ Return a sparse dataframe with columns: item_id  |  feature1 | feature2 | feature3 | feature4... """
    # one-hot encoding of the accomodations features
    item_ids = accomodations_df.item_id
    accomodations_df.properties = accomodations_df.properties.str.split('|')
    accomodations_df.properties = accomodations_df.properties.fillna('')

    mlb = MultiLabelBinarizer(sparse_output=sparse)
    attributes_df = mlb.fit_transform(accomodations_df.properties)
    if sparse:
        attributes_df = pd.SparseDataFrame(attributes_df, columns=mlb.classes_)
    else:
        attributes_df = pd.DataFrame(attributes_df, columns=mlb.classes_)

    # re-add item_id column to get: item_id  |  one_hot_features[...]
    attributes_df = pd.concat([item_ids, attributes_df], axis=1)
    return attributes_df

def scale_dataframe(df, columns_to_scale, fill_nan='0'):
    scaler = MinMaxScaler()
    for col in columns_to_scale:
        df[col] = scaler.fit_transform(df[col].fillna(value=fill_nan).values)
    return df

def df2vec(df, accomodations_df, sparse):
    if sparse:
        df = df.to_sparse()
    # one-hot encoding of the accomodations features
    attributes_df = get_accomodations_one_hot(accomodations_df, sparse=sparse)
    features_columns_count = attributes_df.shape[1] -1

    # substitute a dataframe column with its one-hot encoding derived columns
    def one_hot_df_column(df, column_label, sparse):
        return pd.concat([df, pd.get_dummies(df[column_label], prefix=column_label, sparse=sparse)], axis=1).drop([column_label],axis=1)

    # add the one-hot encoding of the action_type feature
    df = one_hot_df_column(df, 'action_type', sparse)

    # add the one-hot encoding of the device feature
    df = one_hot_df_column(df, 'device', sparse)

    # add the 'no-reference' column
    #df['no_reference'] = (~df.reference.fillna('').str.isnumeric()) * 1
    
    # add the one-hot encoding columns of the reference accomodation
    #old_references = df.references
    # set to nan the non numeric references to allow join on indices
    #df.loc[df.reference.str.isnumeric() != True, 'reference'] = np.nan

    print('Joining accomodations features...')
    df = df.astype({'reference':str}).reset_index().merge(attributes_df.astype({'item_id':str}), how='left', left_on='reference',right_on='item_id').set_index('index')
    print('Done!')
    
    # set all the one-hot features to 0 for the non-reference rows
    accomodations_features_columns = attributes_df.drop('item_id', axis=1).columns
    df[accomodations_features_columns] = df[accomodations_features_columns].fillna(value=0).astype(int)

    return df, df.columns[-features_columns_count:], attributes_df


def get_targets(sess2vec_df, target_columns):
    """
    Get the target vectors: features of the clickout references.
    For the clickout interactions, the target vector is the reference fearures.
    For the non-clickout interactions, the target vector is all zeros.
    """
    targets = sess2vec_df[target_columns]
    # set to 0 the non-clickout interactions
    targets.loc[ (sess2vec_df['action_type_clickout item'] == 0), : ] = 0
    return targets


def add_impressions_columns_as_new_actions(df, sparse):
    df['impression_price'] = np.nan
    clickout_rows = df[df.action_type == 'clickout item']
    print('Total clickout interactions found:', clickout_rows.shape[0])
    
    temp_df = []
    
    for _, row in tqdm(clickout_rows.iterrows()):
        impressions = list(map(int, row.impressions.split('|')))
        prices = list(map(int, row.prices.split('|')))
        row.action_type = 'show_impression'
  
        steps = np.linspace(row.step-1+1/len(impressions),row.step,len(impressions)+1)
        for imprsn, impr_price, step in zip(impressions, prices, steps):
            row.reference = imprsn
            row.impression_price = impr_price
            row.step = step
            #row.original_index = -1   # the show_impression-type rows indices is -1
            temp_df.append(tuple(row))

    if sparse:
        temp_df = pd.SparseDataFrame(temp_df, columns=clickout_rows.columns)
    else:
        temp_df = pd.DataFrame(temp_df, columns=clickout_rows.columns)
    df = df.append(temp_df).drop(['impressions','prices'], axis=1)
    df.index = df.index.set_names(['index'])
    
    return df.sort_values(['user_id','session_id','timestamp','step'])


def create_dataset(train_df, test_df, path, sparse):
    accomodations_df = data.accomodations_df()
    max_train_index = train_df.index[-1]

    full_df = pd.concat([train_df, test_df])

    # add the impressions as new actions
    print('Adding impressions as new actions...')
    full_df = add_impressions_columns_as_new_actions(full_df, sparse=sparse)
    print('Done!')

    print('One-hot encoding the dataset...')
    full_df, target_cols, accomodations_onehot_df = df2vec(full_df, accomodations_df, sparse=False)
    target_df = get_targets(full_df, target_cols)
    print('Done!')

    # resplit train and test, finding the last-index location of the train df
    iloc_of_last_train_interaction = np.where(full_df.index.values == max_train_index)[0][0] + 1
    train_df = full_df.iloc[0:iloc_of_last_train_interaction]
    train_targets_df = target_df.iloc[0:iloc_of_last_train_interaction]
    
    test_df = full_df.iloc[iloc_of_last_train_interaction:]
    test_targets_df = target_df.iloc[iloc_of_last_train_interaction:]

    # save the accomodations df one-hot for later usage
    print('Saving one-hot encoded accomodations...', end=' ', flush=True)
    accomodations_onehot_df.to_csv('dataset/preprocessed/accomodations_vec.csv',index=False)
    print('Done!')
    
    print('Saving train...', end=' ', flush=True)
    train_df.to_csv( os.path.join(path, 'train.csv'), float_format='%.4f' )
    train_targets_df.to_csv( os.path.join(path, 'train_target.csv') )
    print('Done!')

    print('Saving train...', end=' ', flush=True)
    test_df.to_csv( os.path.join(path, 'test.csv'), float_format='%.4f' )
    test_targets_df.to_csv( os.path.join(path, 'test_target.csv') )
    print('Done!')

# ======== POST-PROCESSING ========= #

def load_and_prepare_dataset(mode):
    """ Load the one-hot dataset and return X_train, Y_train, X_test, Y_test """
    train_df = pd.read_csv(f'dataset/preprocessed/cluster_recurrent/{mode}_vec/train.csv').set_index('index')
    train_target_df = pd.read_csv(f'dataset/preprocessed/cluster_recurrent/{mode}_vec/train_target.csv').set_index('index')
    test_df = pd.read_csv(f'dataset/preprocessed/cluster_recurrent/{mode}_vec/test.csv').set_index('index')
    test_target_df = pd.read_csv(f'dataset/preprocessed/cluster_recurrent/{mode}_vec/test_target.csv').set_index('index')

    # drop currently unused columns
    col_to_drop = ['timestamp','step','reference','platform','city','current_filters','item_id']
    X_train = train_df.drop(col_to_drop, axis=1)
    X_test = test_df.drop(col_to_drop, axis=1)

    X_train.impression_price = X_train.impression_price.fillna(value=0)
    X_test.impression_price = X_test.impression_price.fillna(value=0)
    
    return X_train, train_target_df, X_test, test_target_df


def get_session_groups_indices_df(X_df, Y_df, cols_to_group=['user_id','session_id'], indices_col_name='intrctns_indices'):
    """
    Return a dataframe with columns 'user_id', 'session_id', 'intrctns_indices'. This is used to retrieve the
    interaction indices from a particular session id of a user
    """
    return X_df.groupby(cols_to_group).apply(lambda r: pd.Series({indices_col_name: r.index.values})).reset_index()



if __name__ == "__main__":
    import utils.menu as menu
    from utils.check_folder import check_folder

    mode = menu.mode_selection()
    
    train_df = data.train_df(mode, cluster='cluster_recurrent')
    test_df = data.test_df(mode, cluster='cluster_recurrent')

    folder_path = f'dataset/preprocessed/cluster_recurrent/{mode}_vec'
    check_folder(folder_path)

    create_dataset(train_df, test_df, folder_path, sparse=False)