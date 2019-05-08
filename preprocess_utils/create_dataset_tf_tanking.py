import data
from functools import reduce
import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import utils.check_folder as cf

from extract_features.actions_involving_impression_session import ActionsInvolvingImpressionSession
from extract_features.frenzy_factor_consecutive_steps import FrenzyFactorSession
from extract_features.global_clickout_popularity import GlobalClickoutPopularity
from extract_features.global_interactions_popularity import GlobalInteractionsPopularity
from extract_features.impression_features import ImpressionFeature
from extract_features.impression_position_session import ImpressionPositionSession
from extract_features.impression_price_info_session import ImpressionPriceInfoSession
from extract_features.label import ImpressionLabel
from extract_features.last_action_involving_impression import LastInteractionInvolvingImpression
from extract_features.mean_price_clickout import MeanPriceClickout
from extract_features.price_position_info_interactions import PricePositionInfoInteractedReferences
#from extract_features.session_actions_num_ref_diff_from_impressions import SessionActionNumRefDiffFromImpressions
from extract_features.session_device import SessionDevice
from extract_features.session_filters_active_when_clickout import SessionFilterActiveWhenClickout
from extract_features.session_length import SessionLength
from extract_features.session_sort_order_when_clickout import SessionSortOrderWhenClickout
from extract_features.time_from_last_action_before_clk import TimeFromLastActionBeforeClk
from extract_features.times_impression_appeared_in_clickouts_session import TimesImpressionAppearedInClickoutsSession
from extract_features.times_user_interacted_with_impression import TimesUserInteractedWithImpression
from extract_features.timing_from_last_interaction_impression import TimingFromLastInteractionImpression

def is_target(df, tgt_usersession):
    if tuple(df.head(1)[['user_id', 'session_id']].values[0]) in tgt_usersession:
        return True
    else:
        return False

def _reinsert_clickout(df):
    # take the row of the missing clickout
    clickout_rows_df = df[(df['action_type'] == 'clickout item') & df['reference'].isnull()]
    # check if it exsists
    if len(clickout_rows_df)>0:
        # retrieve from the full_df the clickout
        missing_click = data.full_df().loc[clickout_rows_df.index[0]]['reference']
        # reinsert the clickout on the df
        df.at[clickout_rows_df.index[0], 'reference']= missing_click
    return df

def merge_features(mode, cluster, features_array, popularity=False):
    """
    RETRIEVE THE FEATURES
    """

    # load the first feature df
    df_merged = features_array[0](mode=mode, cluster=cluster).read_feature(one_hot=True)
    for f in features_array[1:]:
        print(df_merged.shape)
        df_merged = f(mode=mode, cluster=cluster).join_to(df_merged, one_hot=True)

    print(df_merged.shape)

    if popularity:
        f1 = GlobalClickoutPopularity().read_feature()
        df_merged=pd.merge(df_merged, f1, left_on='item_id', right_on='reference', how='left')
        df_merged.drop('reference', axis=1, inplace=True)
        print(df_merged.shape)

        f1 = GlobalInteractionsPopularity().read_feature()
        df_merged = pd.merge(df_merged, f1, left_on='item_id', right_on='reference', how='left')
        df_merged.drop('reference', axis=1, inplace=True)
        print(df_merged.shape)

        df_merged = df_merged.fillna(0)
    ################################################

    # load the target indeces of the mode
    target_indeces = data.target_indices(mode, cluster)
    print(f'number of tgt index: {len(target_indeces)}')

    # load the full df
    full_df = pd.read_csv(data.FULL_PATH, usecols=['user_id', 'session_id'])

    # dict that has as keys the couples (user_id, session_id) that are target
    tgt_usersession = {}
    for index in target_indeces:
        tgt_usersession[tuple(full_df.iloc[index][['user_id', 'session_id']].values)] = index

    is_target_ = df_merged.groupby(['user_id', 'session_id']).progress_apply(is_target, tgt_usersession=tgt_usersession)
    df_merged = pd.merge(df_merged, is_target_.reset_index(), on=['user_id', 'session_id'])

    test_df = df_merged[df_merged[0] == True]
    train_df = df_merged[df_merged[0] == False]

    del df_merged

    train_df.drop(columns=[0], inplace=True)
    test_df.drop(columns=[0], inplace=True)

    del full_df

    # retrieve the target indeces in the right order
    couples_dict = {}
    couples_arr = test_df[['user_id', 'session_id']].values
    for c in couples_arr:
        if tuple(c) not in couples_dict:
            couples_dict[tuple(c)] = 1

    target_us_reordered = list(couples_dict.keys())

    target_indeces_reordered = []
    for k in target_us_reordered:
        target_indeces_reordered.append(tgt_usersession[k])

    print(f'number of tgt index: {len(target_indeces_reordered)}')
    target_indeces_reordered = np.array(target_indeces_reordered)

    return train_df, test_df, target_indeces_reordered

def create_dataset(mode, cluster, features_array, dataset_name, popularity):
    _SAVE_BASE_PATH = f'dataset/preprocessed/tf_ranking/{cluster}/{mode}/{dataset_name}'
    cf.check_folder(_SAVE_BASE_PATH)
    train_df, test_df, target_indeces_reordered = merge_features(mode, cluster, features_array, popularity)
    
    """
        CREATE DATA FOR TRAIN

        """
    # associate to each session a QID
    qid = []

    count = 0
    actual_sid = 'culo'
    session_ids = train_df['session_id'].values
    for sid in session_ids:
        if sid != actual_sid:
            actual_sid = sid
            count += 1
        qid.append(count)
    np_qid_train = np.array(qid)

    # the 5 column is the label
    X, Y = train_df.drop(['session_id','user_id','label','item_id'], axis=1), train_df['label']

    del train_df
    scaler = MinMaxScaler(copy=False)
    # normalize the values
    X=scaler.fit_transform(X)
    Y_norm = Y.values
    del Y

    X_train, X_val, Y_train, Y_val, qid_train, qid_val = \
        train_test_split(X, Y_norm, np_qid_train, test_size=0.2, shuffle=False)

    print('SAVING TRAIN DATA...')
    dump_svmlight_file(X_train, Y_train, f'{_SAVE_BASE_PATH}/train.txt', query_id=qid_train, zero_based=False)
    print('DONE')

    print('SAVING VALI DATA...')
    dump_svmlight_file(X_val, Y_val, f'{_SAVE_BASE_PATH}/vali.txt', query_id=qid_val, zero_based=False)
    print('DONE')
    del X_train, X_val, Y_train, Y_val

    """
    CREATE DATA FOR TEST

    """
    # do it also fot the test data
    qid_test = []
    #count = 0 don't reset it
    actual_sid = 'culo'
    session_ids = test_df['session_id'].values
    for sid in session_ids:
        if sid != actual_sid:
            actual_sid = sid
            count += 1
        qid_test.append(count)
    np_qid_test = np.array(qid_test)
    print(np_qid_test)

    if mode != 'full':
        X_test, Y_test = test_df.drop(['session_id','user_id','label','item_id'], axis=1), test_df['label']
        del test_df
        X_test=scaler.fit_transform(X_test)
        Y_test_norm = Y_test.values
        # dummy_label = np.zeros(len(X_test),dtype=np.int)

        print('SAVING TEST DATA...')
        dump_svmlight_file(X_test, Y_test_norm, f'{_SAVE_BASE_PATH}/test.txt', query_id=np_qid_test,
                           zero_based=False)
        print('DONE')

        print('SAVING TARGET_INDICES...')
        np.save(f'{_SAVE_BASE_PATH}/target_indices', target_indeces_reordered)
        print('DONE')
        print('PROCEDURE ENDED CORRECTLY')
    else:
        print('I KNOW IM FULL ;)')
        X_test = test_df.drop(['session_id','user_id','label','item_id'], axis=1)
        del test_df
        X_test=scaler.fit_transform(X_test)
        dummy_label = np.zeros(len(X_test), dtype=np.int)

        print('SAVING TEST DATA...')
        dump_svmlight_file(X_test, dummy_label, f'{_SAVE_BASE_PATH}/test.txt', query_id=np_qid_test,
                           zero_based=False)
        print('DONE')

        print('SAVING TARGET_INDICES...')
        np.save(f'{_SAVE_BASE_PATH}/target_indices', target_indeces_reordered)
        print('DONE')
        print('PROCEDURE ENDED CORRECTLY')

if __name__ == '__main__':
    mode = 'small'
    cluster = 'no_cluster'
    dataset_name = 'no_pos'

    features_array = [ActionsInvolvingImpressionSession, ImpressionLabel, ImpressionPriceInfoSession,
                      TimingFromLastInteractionImpression, TimesUserInteractedWithImpression,
                      LastInteractionInvolvingImpression,
                      TimesImpressionAppearedInClickoutsSession, MeanPriceClickout, SessionLength,
                      TimeFromLastActionBeforeClk, PricePositionInfoInteractedReferences,
                      SessionDevice, SessionFilterActiveWhenClickout, SessionSortOrderWhenClickout,
                      ImpressionFeature]

    """
    features_array = {
        'user_id_session_id_item_id': [ActionsInvolvingImpressionSession, ImpressionLabel, ImpressionPriceInfoSession,
                                       TimingFromLastInteractionImpression, TimesUserInteractedWithImpression,
                                       ImpressionPositionSession, LastInteractionInvolvingImpression,
                                       TimesImpressionAppearedInClickoutsSession],
        'user_id_session_id': [MeanPriceClickout, SessionLength, TimeFromLastActionBeforeClk,
                               FrenzyFactorSession, PricePositionInfoInteractedReferences,
                               SessionDevice, SessionFilterActiveWhenClickout, SessionSortOrderWhenClickout],
        'item_id': [ImpressionFeature, GlobalInteractionsPopularity, GlobalClickoutPopularity]
    }
    """

    """
    features_array = {
        'item_id': [ImpressionLabel,ImpressionPriceInfoSession,LastInteractionInvolvingImpression,
                    TimingFromLastInteractionImpression,ActionsInvolvingImpressionSession,ImpressionPositionSession,
                    TimesUserInteractedWithImpression,ItemPopularitySession],
        'session': [MeanPriceClickout, MeanPriceClickout_edo, SessionLength, SessionDevice,
                    SessionActionNumRefDiffFromImpressions, SessionFilterActiveWhenClickout,
                    SessionSortOrderWhenClickout, TimePassedBeforeClickout]
    }
    """

    create_dataset(mode=mode, cluster=cluster, features_array=features_array, dataset_name=dataset_name, popularity=True)