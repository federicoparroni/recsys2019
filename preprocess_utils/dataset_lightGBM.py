from scipy.sparse import save_npz
import data
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.datasets import dump_svmlight_file
import pickle
from utils.check_folder import check_folder
from preprocess_utils.merge_features import merge_features
from extract_features.actions_involving_impression_session import ActionsInvolvingImpressionSession
from extract_features.frenzy_factor_consecutive_steps import FrenzyFactorSession
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

import os
from pathlib import Path


def to_queries_dataset(t, save_dataset=True, path=''):
    print('Creating query-based dataset...')
    t = t.sort_values(by=['index', 'impression_position'])
    #t = t.assign(id=(t['user_id'] + '_' + t['session_id']).astype('category').cat.codes)

    # set label column as first column
    cols = list(t.columns)
    cols.insert(0, cols.pop(cols.index('label')))
    t = t.loc[:, cols]
    t = t.drop(['user_id', 'session_id'], axis=1)

    # extract num_rows for each query id: count element in each session
    size = t[['index', 'label']].groupby('index').agg('count')
    c1 = size.index.values
    c2 = size['label'].values
    d = {'id': c1, 'num_rows': c2}
    size_df = pd.DataFrame(d)
    num_rows_df = size_df.drop(['id'], axis=1)

    if save_dataset == True:
        dump_svmlight_file(t.iloc[:, 2:].values, t.iloc[:, 0], path, query_id=t['index'].values)
        num_rows_df.to_csv(str(path) + '.query', header=None, index=None)

    print('Query-based dataset created.')

def create_dataset(mode, cluster):
    # training
    features_array = [ActionsInvolvingImpressionSession, ImpressionLabel, ImpressionPriceInfoSession,
                      TimingFromLastInteractionImpression, TimesUserInteractedWithImpression,
                      ImpressionPositionSession, LastInteractionInvolvingImpression,
                      TimesImpressionAppearedInClickoutsSession, MeanPriceClickout, SessionLength,
                      TimeFromLastActionBeforeClk, FrenzyFactorSession, PricePositionInfoInteractedReferences,
                      SessionDevice, SessionFilterActiveWhenClickout, SessionSortOrderWhenClickout,
                      ImpressionFeature]

    curr_dir = Path(__file__).absolute().parent
    data_dir = curr_dir.joinpath('..', 'dataset/preprocessed/{}/{}/lightGBM/'.format(cluster, mode))
    print(data_dir)
    check_folder(str(data_dir))

    train_df, test_df= merge_features(mode, cluster, features_array)

    if os.path.isfile(str(data_dir) + '/svmlight_train.txt'):
        print('Train File già presente')
    else:
        to_queries_dataset(train_df, path=str(data_dir) + '/svmlight_train.txt')

    if os.path.isfile(str(data_dir) + '/test.csv'):
        print('Test File già presente')
        #test_df.sort_values()
        to_queries_dataset(test_df, path=str(data_dir) + '/svmlight_test.txt')
    else:
        test_df.to_csv(str(data_dir) + '/test.csv', index=False)
        to_queries_dataset(test_df, path=str(data_dir) + '/svmlight_test.txt')


if __name__ == "__main__":
    #from utils.menu import mode_selection
    #mode = mode_selection()
    create_dataset(mode='small', cluster='no_cluster')
