import sys
import os
sys.path.append(os.getcwd())

import data
import utils.menu as menu
import numpy as np
import utils.df as df_utils
import preprocess_utils.sessions_to_predict as sess2predict
from utils.check_folder import check_folder
import utils.datasetconfig as datasetconfig

from clusterize.cluster_recurrent import ClusterRecurrent
from clusterize.cluster_up_to_len6 import ClusterUpToLen6
from clusterize.cluster_over_len6 import ClusterOverLen6

from extract_features.rnn.reference_position_in_last_clickout_impressions import ReferencePositionInLastClickoutImpressions
from extract_features.rnn.reference_price_in_last_clickout import ReferencePriceInLastClickout
from extract_features.rnn.clickout_vector_prices import ClickoutVectorPrices
from extract_features.rnn.reference_price_position_in_last_clickout import ReferencePricePositionInLastClickout
#from extract_features.rnn.global_interactions_popularity import GlobalInteractionsPopularity
from extract_features.rnn.global_clickout_popularity import GlobalClickoutPopularity
from extract_features.rnn.session_impressions_count import SessionsImpressionsCount
from extract_features.rnn.interaction_duration import InteractionDuration
from extract_features.rnn.clickout_filters_satisfaction import ClickoutFiltersSatisfaction
from extract_features.rnn.impressions_popularity import ImpressionsPopularity
from extract_features.rnn.change_sort_order_filters import ChangeSortOrderFilters

import preprocess_utils.session2vec as sess2vec


def create_dataset_for_classification(mode, cluster, pad_sessions_length, add_item_features, add_dummy_actions=False,
                                    features=[], only_test=False, resample=False, one_target_per_session=True):
    """
    pad_sessions_length (int): final length of sessions after padding/truncating
    add_item_features (bool): whether to add the accomodations features as additional columns
    add_dummy_actions (bool): whether to add dummy interactions representing the impressions before each clickout
    features (list): list of classes (inheriting from FeatureBase) that will provide additional features to be joined
    only_test (bool): whether to create only the test dataset (useful to make predictions with a pre-trained model)
    """
    
    path = f'dataset/preprocessed/{cluster}/{mode}/dataset_classification_v2_p{pad_sessions_length}'
    check_folder(path)

    def create_ds_class(df, path, for_train, add_dummy_actions=add_dummy_actions, pad_sessions_length=pad_sessions_length, 
                        add_item_features=add_item_features, resample=resample, one_target_per_session=one_target_per_session,
                        new_row_index=99000000):
        """ Create X and Y dataframes if for_train, else only X dataframe.
            Return the number of rows of the new dataframe and the final index
        """

        ds_type = 'train' if for_train else 'test'
        devices_classes = ['mobile', 'desktop', 'tablet']
        # actions_classes = ['clickout item', 'interaction item rating', 'interaction item info',
        #         'interaction item image', 'interaction item deals', 'search for item', 'search for destination',
        #         'search for poi'] #, 'change of sort order', 'filter selection', 'show_impression', ]
        
        # merge the features
        print('Merging the features...')
        for f in features:
            df = f.join_to(df)
        print('Done!\n')
    
        # add the impressions as new interactions
        if add_dummy_actions:
            print('Adding impressions as new actions...')
            df, new_row_index = sess2vec.add_impressions_as_new_actions(df, drop_cols=['prices'], new_rows_starting_index=new_row_index)
            print('Done!\n')
        else:
            df = df.drop('prices', axis=1)

        # pad the sessions
        if pad_sessions_length > 0:
            print('Padding/truncating sessions...')
            df = sess2vec.pad_sessions(df, max_session_length=pad_sessions_length)
            print('Done!\n')
    
        # print('Getting the last clickout of each session...')
        # print('Done!\n')

        # add the one-hot of the device
        # df = df.drop('device', axis=1)
        print('Adding one-hot columns of device...', end=' ', flush=True)
        df = sess2vec.one_hot_df_column(df, 'device', classes=devices_classes)
        print('Done!\n')

        # add the encoding of the action-type
        print('Adding encoding of action_type...', end=' ', flush=True)
        df = sess2vec.add_actions_custom_encoding(df)
        df = df.drop('action_type', axis=1)
        print('Done!\n')

        # merge interaction focus and change-of-sort-order filters 
        df = sess2vec.aggregate_action_type_and_sof(df)

        # remove the impressions column
        df = df.drop('impressions', axis=1)

        if for_train and resample:
            # resample the dataset to balance the classes
            resample_perc = 0.5 / df.ref_class.mean()
            print('resample perc:', resample_perc)
            df = df_utils.resample_sessions(df, by=resample_perc, when=df_utils.ref_class_is_1)

        # join the accomodations one-hot features
        if add_item_features:
            print('Adding accomodations features...')
            df = sess2vec.merge_reference_features(df, pad_sessions_length)

        X_LEN = df.shape[0]

        # save the X dataframe without the labels (reference classes)
        x_path = os.path.join(path, 'X_{}.csv'.format(ds_type))
        print('Saving X {}...'.format(ds_type), end=' ', flush=True)
        df.to_csv(x_path, index_label='orig_index', float_format='%.4f')
        print('Done!\n')

        if for_train:
            # set the columns to be placed in the Y file
            Y_COLUMNS = ['user_id','session_id','timestamp','step']
            df = df[Y_COLUMNS]

            # take only the target rows from y
            if one_target_per_session:
                df = df.iloc[np.arange(-1,len(df),pad_sessions_length)[1:]]
            
            # add the reference classes
            print('Adding references classes...')
            df = sess2vec.add_reference_labels(df, mode=mode)
            print('Done!\n')

            # save the Y dataframe
            y_path = os.path.join(path, 'Y_train.csv')
            print('Saving Y_train...', end=' ', flush=True)
            df.to_csv(y_path, index_label='orig_index', float_format='%.4f')
            print('Done!\n')
            
        return X_LEN, new_row_index

    final_new_index = 99000000
    TRAIN_LEN = 0
    
    # remove the sessions to be not predicted and move into the train
    test_df = data.test_df(mode, cluster)
    print('Moving sessions not to be predicted from test to train...')
    test_df, sessions_not_to_predict = sess2predict.find(test_df)
    test_df = test_df.sort_values(['user_id','session_id','timestamp','step'])

    if not only_test:
        ## ======== TRAIN ======== ##
        train_df = data.train_df(mode, cluster)
        train_df = train_df.append(sessions_not_to_predict).sort_values(['user_id','session_id','timestamp','step'])
        TRAIN_LEN, final_new_index = create_ds_class(train_df, path, for_train=True, new_row_index=final_new_index)
        del train_df

    ## ======== TEST ======== ##
    TEST_LEN, _ = create_ds_class(test_df, path, for_train=False, new_row_index=final_new_index)
    del test_df

    ## ======== CONFIG ======== ##
    # save the dataset config file that stores dataset length and the list of sparse columns
    #x_sparse_cols = devices_classes + actions_classes
    datasetconfig.save_config(path, mode, cluster, TRAIN_LEN, TEST_LEN,
                                rows_per_sample=pad_sessions_length)
                            #X_sparse_cols=x_sparse_cols, Y_sparse_cols=ref_classes)



if __name__ == "__main__":
        
    mode = menu.mode_selection()
    #cluster_name = 'cluster_recurrent'
    cluster = menu.single_choice('Which cluster?', ['cluster recurrent','cluster len <= 6', 'cluster len > 6'],
                                    callbacks=[lambda: ClusterRecurrent, lambda: ClusterUpToLen6, lambda: ClusterOverLen6])
    c = cluster()

    # create the cluster
    cluster_choice = menu.yesno_choice('Do you want to create the cluster?', lambda: True, lambda: False)
    if cluster_choice:
        c.save(mode)
        print()

    only_test = False
    if mode != 'small':
        only_test = menu.yesno_choice('Do you want to create only the test dataset?', lambda: True, lambda: False)
    
    sess_length = int(input('Insert the desired sessions length, -1 to not to pad/truncate the sessions: '))

    features_to_join = [
        #ReferencePositionInNextClickoutImpressions,
        ReferencePositionInLastClickoutImpressions,
        GlobalClickoutPopularity,
        #GlobalInteractionsPopularity,
        #AveragePriceInNextClickout,
        
        #ReferencePriceInNextClickout,
        ReferencePriceInLastClickout,
        ClickoutVectorPrices,

        #ReferencePricePositionInNextClickout,
        ReferencePricePositionInLastClickout,

        SessionsImpressionsCount,

        InteractionDuration,

        ClickoutFiltersSatisfaction,
        ImpressionsPopularity,

        ChangeSortOrderFilters
    ]
    features = []
    # create the features to join
    for f in features_to_join:
        feat = f(mode, c.name)
        feat.save_feature()
        features.append(feat)

    # create the tensors dataset
    print('Creating the dataset ({})...'.format(mode))
    create_dataset_for_classification(mode, c.name, pad_sessions_length=sess_length, resample=False,
                                        add_item_features=False, features=features, only_test=only_test)


