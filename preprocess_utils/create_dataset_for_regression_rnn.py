import sys
import os
sys.path.append(os.getcwd())

import data
import utils.menu as menu
from utils.check_folder import check_folder
import utils.datasetconfig as datasetconfig
import numpy as np

from clusterize.cluster_recurrent import ClusterRecurrent
from clusterize.cluster_recurrent_up_to_len6 import ClusterRecurrentUpToLen6

from extract_features.reference_position_in_next_clickout_impressions import ReferencePositionInNextClickoutImpressions
from extract_features.global_interactions_popularity import GlobalInteractionsPopularity
from extract_features.global_clickout_popularity import GlobalClickoutPopularity
from extract_features.reference_price_in_next_clickout import ReferencePriceInNextClickout
from extract_features.average_price_in_next_clickout import AveragePriceInNextClickout
from extract_features.reference_price_position_in_next_clickout import ReferencePricePositionInNextClickout

import preprocess_utils.session2vec as sess2vec


def create_dataset_for_regression(mode, cluster, pad_sessions_length, add_item_features=True, save_X_Y=True):
    """
    pad_sessions_length (int): final length of sessions after padding/truncating
    add_item_features (bool): whether to add the one-hot accomodations features to the training data
    save_X_Y (bool): whether to save the train data into 2 separate files (X_train, Y_train) or in a unique file (train_vec)
    """
    train_df = data.train_df(mode, cluster)
    test_df = data.test_df(mode, cluster)

    path = f'dataset/preprocessed/{cluster}/{mode}/dataset_regression'
    check_folder(path)

    devices_classes = ['mobile', 'desktop', 'tablet']
    actions_classes = ['show_impression', 'clickout item', 'interaction item rating', 'interaction item info',
           'interaction item image', 'interaction item deals', 'change of sort order', 'filter selection',
           'search for item', 'search for destination', 'search for poi']
    
    ## ======== TRAIN ======== ##
    # add the impressions as new interactions
    print('Adding impressions as new actions...')
    train_df, final_new_index = sess2vec.add_impressions_as_new_actions(train_df)
    print('Done!\n')

    # pad the sessions
    if pad_sessions_length > 0:
        print('Padding/truncating sessions...')
        train_df = sess2vec.pad_sessions(train_df, max_session_length=pad_sessions_length)
        print('Done!\n')

    if save_X_Y:
        print('Getting the last clickout of each session...')
        train_clickouts_df = sess2vec.get_last_clickout(train_df, index_name='index', rename_index='orig_index')
        train_clickouts_indices = train_clickouts_df.orig_index.values
        train_clickouts_indices.sort()
        print('Done!\n')

    # add the one-hot of the device
    print('Adding one-hot columns of device...', end=' ', flush=True)
    train_df = sess2vec.one_hot_df_column(train_df, 'device', classes=devices_classes)
    print('Done!\n')

    # add the one-hot of the action-type
    print('Adding one-hot columns of action_type...', end=' ', flush=True)
    train_df = sess2vec.one_hot_df_column(train_df, 'action_type', classes=actions_classes)
    print('Done!\n')

    TRAIN_LEN = train_df.shape[0]
    TRAIN_NAME = ''

    if save_X_Y:
        # set the columns to be placed in the labels file
        Y_COLUMNS = ['user_id','session_id','timestamp','step','reference']
    
        # join the accomodations one-hot features
        X_train_path = os.path.join(path, 'X_train.csv')
        if add_item_features:
            print('Joining the accomodations features...')
            sess2vec.add_accomodations_features(train_df.copy(), X_train_path, logic='skip', row_indices=train_clickouts_indices)
        else:
            # set the last clickouts to NaN and save the X dataframe
            backup_ref_serie = train_df.reference.values.copy()
            train_df.loc[train_clickouts_indices, 'reference'] = np.nan
            train_df.to_csv(X_train_path, index_label='orig_index', float_format='%.4f')
            train_df.reference = backup_ref_serie
            del backup_ref_serie
        
        Y_train_path = os.path.join(path, 'Y_train.csv')
        train_df = train_df[Y_COLUMNS]
        if add_item_features:
            sess2vec.add_accomodations_features(train_df.copy(), Y_train_path, logic='subset', row_indices=train_clickouts_indices)
        else:
            # set all clickouts to NaN except for the last clickouts and save the Y dataframe
            backup_ref_serie = train_df.loc[train_clickouts_indices].reference.copy()
            train_df.reference = np.nan
            train_df.loc[train_clickouts_indices, 'reference'] = backup_ref_serie
            train_df.to_csv(Y_train_path, index_label='orig_index', float_format='%.4f')

        # clean ram
        del train_clickouts_df
        del train_clickouts_indices
    else:
        TRAIN_NAME = 'train_vec.csv'
        train_path = os.path.join(path, TRAIN_NAME)
        train_df.to_csv(train_path, index_label='orig_index', float_format='%.4f')

    del train_df

    ## ======== TEST ======== ##
    print('Adding impressions as new actions...')
    test_df, _ = sess2vec.add_impressions_as_new_actions(test_df, final_new_index)
    print('Done!\n')

    # pad the sessions
    if pad_sessions_length > 0:
        print('Padding/truncating sessions...')
        test_df = sess2vec.pad_sessions(test_df, max_session_length=pad_sessions_length)
        print('Done!\n')

    print('Getting the last clickout of each session...')
    test_clickouts_df = sess2vec.get_last_clickout(test_df, index_name='index', rename_index='orig_index')
    test_clickouts_indices = test_clickouts_df.orig_index.values
    test_clickouts_indices.sort()
    print('Done!\n')

    # add the one-hot of the device
    print('Adding one-hot columns of device...', end=' ', flush=True)
    test_df = sess2vec.one_hot_df_column(test_df, 'device', classes=devices_classes)
    print('Done!\n')

    # add the one-hot of the action-type
    print('Adding one-hot columns of action_type...', end=' ', flush=True)
    test_df = sess2vec.one_hot_df_column(test_df, 'action_type', classes=actions_classes)
    print('Done!\n')

    TEST_LEN = test_df.shape[0]

    # join the accomodations one-hot features
    X_test_path = os.path.join(path, 'X_test.csv')
    if add_item_features:
        print('Joining the accomodations features...')
        sess2vec.add_accomodations_features(test_df.copy(), X_test_path, logic='skip', row_indices=test_clickouts_indices)
    else:
        # set the last clickouts to NaN and save the X dataframe
        backup_ref_serie = test_df.reference.values.copy()
        test_df.loc[test_clickouts_indices, 'reference'] = np.nan
        test_df.to_csv(X_test_path, index_label='orig_index', float_format='%.4f')
        #test_df.reference = backup_ref_serie
        del backup_ref_serie
    
    ## ======== CONFIG ======== ##
    # save the dataset config file that stores dataset length and the list of sparse columns
    features_cols = list(data.accomodations_one_hot().columns) if add_item_features else []    
    x_sparse_cols = devices_classes + actions_classes + features_cols
    datasetconfig.save_config(path, mode, cluster, TRAIN_LEN, TEST_LEN, train_name=TRAIN_NAME,
                            rows_per_sample=pad_sessions_length,
                            X_sparse_cols=x_sparse_cols, Y_sparse_cols=features_cols)

    



if __name__ == "__main__":
        
    mode = menu.mode_selection()
    #cluster_name = 'cluster_recurrent'
    cluster = menu.single_choice('Which cluster?', ['cluster recurrent','cluster recurrent len <= 6'],
                                    callbacks=[lambda: ClusterRecurrent, lambda: ClusterRecurrentUpToLen6])
    c = cluster()

    # create the cluster
    cluster_choice = menu.yesno_choice('Do you want to create the cluster?', lambda: True, lambda: False)
    if cluster_choice:
        print('Creating the cluster...')
        c.save(mode)
        print()

    only_test = False
    if mode != 'small':
        only_test = menu.yesno_choice('Do you want to create only the test dataset?', lambda: True, lambda: False)
    
    sess_length = int(input('Insert the desired sessions length, -1 to not to pad/truncate the sessions: '))

    features_to_join = [
        ReferencePositionInNextClickoutImpressions,
        GlobalClickoutPopularity,
        GlobalInteractionsPopularity,
        AveragePriceInNextClickout,
        ReferencePriceInNextClickout,
        ReferencePricePositionInNextClickout,
    ]
    features = []
    # create the features to join
    for f in features_to_join:
        feat = f()
        feat.save_feature()
        features.append(feat)
        print()

    # create the tensors dataset
    print('Creating the dataset ({})...'.format(mode))
    create_dataset_for_regression(mode, c.name, pad_sessions_length=sess_length)
                                    #add_item_features=False, features=features, only_test=only_test)


