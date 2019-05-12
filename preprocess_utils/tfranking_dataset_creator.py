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

def find_last_clickout_indices(df):
    indices = []
    cur_ses = ''
    cur_user = ''
    temp_df = df[df.action_type == 'clickout item'][['user_id', 'session_id', 'action_type']]
    for idx in tqdm(temp_df.index.values[::-1]):
        ruid = temp_df.at[idx, 'user_id']
        rsid = temp_df.at[idx, 'session_id']
        if (ruid != cur_user or rsid != cur_ses):
            indices.append(idx)
            cur_user = ruid
            cur_ses = rsid
    return indices[::-1]


def expand_impressions(df):
    res_df = df.copy()
    res_df.impressions = res_df.impressions.str.split('|')
    res_df = res_df.reset_index()

    res_df = pd.DataFrame({
        col: np.repeat(res_df[col].values, res_df.impressions.str.len())
        for col in res_df.columns.drop('impressions')}
    ).assign(**{'impressions': np.concatenate(res_df.impressions.values)})[res_df.columns]

    res_df = res_df.rename(columns={'impressions': 'item_id'})
    res_df = res_df.astype({'item_id':'int'})
    return res_df[['index', 'step','user_id', 'session_id', 'item_id']]

def merge_features(mode, cluster, features_array):
    # load the full_df
    train_df = data.train_df(mode, cluster)
    test_df = data.test_df(mode, cluster)
    full_df = pd.concat([train_df, test_df])
    del train_df, test_df

    # retrieve the indeces of the last clikcouts
    print('find_last_click_idxs')
    last_click_idxs=find_last_clickout_indices(full_df)

    # filter on the found indeces obtaining only the rows of a last clickout
    print('filter full on last click idxs')
    click_df = full_df.loc[last_click_idxs].copy()

    print('retrieve vali_idxs')
    # if the mode is full we don't have the validation if the mode is small or local the validation is performed
    # on the target indices

    vali_test_idxs = data.target_indices(mode, cluster)


    # construct the validation train and test df_base
    print('construct test and vali df')
    validation_test_df = click_df.loc[vali_test_idxs]

    all_idxs = click_df.index.values

    # find the differences
    print('construct train df')
    train_idxs = np.setdiff1d(all_idxs, vali_test_idxs, assume_unique=True)
    train_df = click_df.loc[train_idxs]

    # expand the impression as rows
    print('expand the impression')
    train_df = expand_impressions(train_df)
    validation_test_df = expand_impressions(validation_test_df)

    # do the join
    print('join with the features')
    print(f'train_shape: {train_df.shape}\n vali_test_shape: {validation_test_df.shape}')
    for f in features_array:
        feature = f(mode=mode, cluster=cluster).read_feature(one_hot=True)
        print(f'len of feature:{len(feature)}')
        train_df = train_df.merge(feature)
        validation_test_df = validation_test_df.merge(feature)
        print(f'train_shape: {train_df.shape}\n vali_shape: {validation_test_df.shape}')

    print('sorting by index and step...')
    # sort the dataframes
    train_df.sort_values(['index', 'step'], inplace=True)
    validation_test_df.sort_values(['index', 'step'], inplace=True)

    print('after join')
    return train_df, validation_test_df

def dump_svmlight(df, save_path):
    print(len(df['index'].unique()))
    qid = df['index'].values
    X, Y = df.drop(['session_id', 'user_id', 'label', 'item_id', 'index', 'step'], axis=1), df['label']
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
    train_df, vali_test_df= merge_features(mode, cluster, features_array)

    dump_svmlight(train_df, f'{_SAVE_BASE_PATH}/train.txt')
    if mode == 'full':
        dump_svmlight(vali_test_df, f'{_SAVE_BASE_PATH}/test.txt')
    else:
        dump_svmlight(vali_test_df, f'{_SAVE_BASE_PATH}/vali.txt')
    print('PROCEDURE ENDED CORRECTLY')


if __name__ == '__main__':
    features_array = [ActionsInvolvingImpressionSession, ImpressionLabel, ImpressionPriceInfoSession,
                      TimingFromLastInteractionImpression, TimesUserInteractedWithImpression,
                      ImpressionPositionSession,LastInteractionInvolvingImpression,
                      TimesImpressionAppearedInClickoutsSession, MeanPriceClickout, SessionLength,
                      TimeFromLastActionBeforeClk, PricePositionInfoInteractedReferences,
                      SessionDevice]

    print('inser mode:')
    mode = input()
    cluster = 'no_cluster'
    print('inser dataset_name:')
    dataset_name = input()

    create_dataset(mode, cluster, features_array, dataset_name)
