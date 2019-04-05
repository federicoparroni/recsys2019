import sys
import os
sys.path.append(os.getcwd())

import data
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

def df2vec(df, accomodations_df):
    # one-hot encoding of the accomodations features
    tqdm.pandas()
    accomodations_df.properties = accomodations_df.properties.progress_apply(
        lambda x: x.split('|') if isinstance(x, str) else x)
    accomodations_df.fillna(value='', inplace=True)

    mlb = MultiLabelBinarizer()
    one_hot_features = mlb.fit_transform(accomodations_df.properties)
    one_hot_accomodations_df = pd.DataFrame(one_hot_features, columns=mlb.classes_)

    # create dataframe with columns: item_id  |  one_hot_features[...]
    attributes_df = pd.concat([accomodations_df.drop('properties', axis=1), one_hot_accomodations_df], axis=1)
    features_columns_count = attributes_df.shape[1] -1

    # substitute a dataframe column with its one-hot encoding derived columns
    def one_hot_df_column(df, column_label):
        return pd.concat([df, pd.get_dummies(df[column_label], prefix=column_label)], axis=1).drop([column_label],axis=1)
    
    #print(df.loc[3677].session_id)

    # add the one-hot encoding of the action_type feature
    df = one_hot_df_column(df, 'action_type')
    #print(df.loc[3677].session_id)

    # add the one-hot encoding of the device feature
    df = one_hot_df_column(df, 'device')
    #print(df.loc[3677].session_id)

    # add the 'no-reference' column
    df['no_reference'] = (~df.reference.fillna('').str.isnumeric()) * 1
    
    # add the one-hot encoding columns of the reference accomodation
    df = df.astype({'reference':str}).reset_index().merge(attributes_df.astype({'item_id':str}), how='left', left_on='reference',right_on='item_id').set_index('index')
    #print(df.loc[3677].session_id)

    # set all the one-hot features to 0 for the non-reference rows
    accomodations_features_columns = attributes_df.drop('item_id', axis=1).columns
    df[accomodations_features_columns] = df[accomodations_features_columns].fillna(value=0).astype(int)
    #print(df.loc[3677].session_id)

    return df, df.columns[-features_columns_count:]


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


def create_dataset(train_df, test_df, path):
    accomodations_df = data.accomodations_df()
    train_rows_count = train_df.shape[0]

    full_df = pd.concat([train_df, test_df])

    full_df, target_cols = df2vec(full_df, accomodations_df)
    target_df = get_targets(full_df, target_cols)

    # save train and test
    train_df = full_df.iloc[0:train_rows_count]
    train_targets_df = target_df.iloc[0:train_rows_count]
    
    test_df = full_df.iloc[train_rows_count:]
    test_targets_df = target_df.iloc[train_rows_count:]

    train_df.to_csv( os.path.join(path, 'train.csv') )
    train_targets_df.to_csv( os.path.join(path, 'train_target.csv') )
    test_df.to_csv( os.path.join(path, 'test.csv') )
    test_targets_df.to_csv( os.path.join(path, 'test_target.csv') )


def load_and_prepare_dataset(mode):
    """ Load the one-hot dataset and return X_train, Y_train, X_test, Y_test """
    train_df = pd.read_csv(f'dataset/preprocessed/cluster_recurrent/{mode}_vec/train.csv').set_index('index')
    train_target_df = pd.read_csv(f'dataset/preprocessed/cluster_recurrent/{mode}_vec/train_target.csv').set_index('index')
    test_df = pd.read_csv(f'dataset/preprocessed/cluster_recurrent/{mode}_vec/test.csv').set_index('index')
    test_target_df = pd.read_csv(f'dataset/preprocessed/cluster_recurrent/{mode}_vec/test_target.csv').set_index('index')

    # drop currently unused columns
    col_to_drop = ['timestamp','step','reference','platform','city','current_filters','impressions','prices','item_id']
    X_train = train_df.drop(col_to_drop, axis=1)
    X_test = test_df.drop(col_to_drop, axis=1)
    
    return X_train, train_target_df, X_test, test_target_df


def get_session_groups_indices_df(X_df, Y_df, cols_to_group=['user_id','session_id'], indices_col_name='intrctns_indices'):
    """
    Return a dataframe with columns 'user_id', 'session_id', 'intrctns_indices'. This is used to retrieve the interaction indices
    from a particular session id of a user
    """
    return X_df.groupby(cols_to_group).apply(lambda r: pd.Series({indices_col_name: r.index.values})).reset_index()



if __name__ == "__main__":
    import utils.menu as menu
    import utils.check_folder as check_folder

    mode = menu.mode_selection()
    
    train_df = data.train_df(mode, cluster='cluster_recurrent')
    test_df = data.test_df(mode, cluster='cluster_recurrent')

    folder_path = f'dataset/preprocessed/cluster_recurrent/{mode}_vec'
    check_folder(folder_path)

    create_dataset(train_df, test_df, folder_path)