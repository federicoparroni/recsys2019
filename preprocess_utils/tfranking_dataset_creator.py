import data
from functools import reduce
import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import utils.check_folder as cf

from extract_features.action_type_bef_click import ActionTypeBefClick
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
from extract_features.weights_class import WeightsClass
from preprocess_utils.merge_features import merge_features



def dump_svmlight(df, save_path):
    print(len(df['index'].unique()))
    qid = df['index'].values
    X, Y = df.drop(['session_id', 'user_id', 'label', 'item_id', 'index'], axis=1), df['label']
    del df
    # scaler = MinMaxScaler(feature_range=(-1, 1), copy=False)
    scaler = MaxAbsScaler(copy=False)
    # normalize the values
    X = scaler.fit_transform(X)
    Y_norm = Y.values
    del Y
    print(f'SAVING DATA... \n {save_path}')
    dump_svmlight_file(X, Y_norm, save_path, query_id=qid, zero_based=False)
    print('DONE')


def create_dataset(mode, cluster, features_array, dataset_name):
    _SAVE_BASE_PATH = f'dataset/preprocessed/tf_ranking/{cluster}/{mode}/{dataset_name}'
    cf.check_folder(_SAVE_BASE_PATH)
    train_df, vali_test_df=merge_features(mode, cluster, features_array)

    dump_svmlight(train_df, f'{_SAVE_BASE_PATH}/train.txt')
    if mode == 'full':
        dump_svmlight(vali_test_df, f'{_SAVE_BASE_PATH}/test.txt')
    else:
        dump_svmlight(vali_test_df, f'{_SAVE_BASE_PATH}/vali.txt')
    print('PROCEDURE ENDED CORRECTLY')


if __name__ == '__main__':

    features_array = [ImpressionLabel, ImpressionPositionSession]


    """
    features_array = [ActionsInvolvingImpressionSession, ImpressionLabel, ImpressionPriceInfoSession,
                      TimingFromLastInteractionImpression, TimesUserInteractedWithImpression,
                      ImpressionPositionSession, LastInteractionInvolvingImpression,
                      TimesImpressionAppearedInClickoutsSession, MeanPriceClickout, SessionLength,
                      TimeFromLastActionBeforeClk, PricePositionInfoInteractedReferences,
                      SessionDevice, ActionTypeBefClick]
    """

    print('insert mode:')
    mode = input()
    print('insert cluster name:')
    cluster = input()
    print('insert dataset_name:')
    dataset_name = input()

    create_dataset(mode, cluster, features_array, dataset_name)
