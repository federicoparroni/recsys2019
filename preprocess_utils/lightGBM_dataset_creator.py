import lightgbm as lgb
import data
from tqdm import tqdm
import pandas as pd
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
from preprocess_utils.last_clickout_indices import expand_impressions
import numpy as np

from scipy.sparse import save_npz
import data
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from utils.check_folder import check_folder
from extract_features.impression_rating_numeric import ImpressionRatingNumeric
from extract_features.actions_involving_impression_session import ActionsInvolvingImpressionSession
from extract_features.frenzy_factor_consecutive_steps import FrenzyFactorSession
from extract_features.impression_features import ImpressionFeature
from extract_features.impression_position_session import ImpressionPositionSession
from extract_features.impression_price_info_session import ImpressionPriceInfoSession
from extract_features.label import ImpressionLabel
from extract_features.last_action_involving_impression import LastInteractionInvolvingImpression
from extract_features.mean_price_clickout import MeanPriceClickout
from extract_features.price_position_info_interactions import PricePositionInfoInteractedReferences
# from extract_features.session_actions_num_ref_diff_from_impressions import SessionActionNumRefDiffFromImpressions
from extract_features.session_device import SessionDevice
from extract_features.session_filters_active_when_clickout import SessionFilterActiveWhenClickout
from extract_features.session_length import SessionLength
from extract_features.session_sort_order_when_clickout import SessionSortOrderWhenClickout
from extract_features.time_from_last_action_before_clk import TimeFromLastActionBeforeClk
from extract_features.times_impression_appeared_in_clickouts_session import TimesImpressionAppearedInClickoutsSession
from extract_features.times_user_interacted_with_impression import TimesUserInteractedWithImpression
from extract_features.timing_from_last_interaction_impression import TimingFromLastInteractionImpression
from extract_features.weights_class import WeightsClass
from extract_features.impression_rating import ImpressionRating
from extract_features.time_per_impression import TimeImpressionLabel
from extract_features.session_impression_count_numeric import SessionsImpressionsCountNumeric
from extract_features.action_type_bef_click import ActionTypeBefClick
from extract_features.change_impression_order_position_in_session import ChangeImpressionOrderPositionInSession
from extract_features.session_actions_num_ref_diff_from_impressions import SessionActionNumRefDiffFromImpressions
from extract_features.top_pop_per_impression import TopPopPerImpression
from extract_features.top_pop_interaction_clickout_per_impression import TopPopInteractionClickoutPerImpression
from utils.menu import single_choice
from preprocess_utils.merge_features import merge_features
from os.path import join

def merge_features_lgb(mode, cluster, features_array):

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
    train_df = expand_impressions(train_df)[['user_id', 'session_id', 'item_id', 'index']]
    train_df['dummy_step']=np.arange(len(train_df))
    validation_test_df = expand_impressions(validation_test_df)[['user_id', 'session_id', 'item_id', 'index']]
    validation_test_df['dummy_step'] = np.arange(len(validation_test_df))

    # do the join
    print('join with the features')
    print(f'train_shape: {train_df.shape}\n vali_test_shape: {validation_test_df.shape}')
    categorical_idxs = []
    for f in features_array:
        _feature = f(mode=mode, cluster='no_cluster')
        feature = _feature.read_feature(one_hot=False)
        if len(_feature.columns_to_onehot)>0:
            for i in range(train_df.shape[1]-6, train_df.shape[1]-6+feature.shape[1]-2+1, 1):
                categorical_idxs.append(i)
        print(f'categorical features id:{categorical_idxs}\n')
        print(f'shape of feature: {feature.shape}\n')
        print(f'len of feature:{len(feature)}\n')
        train_df = train_df.merge(feature)
        validation_test_df = validation_test_df.merge(feature)
        print(f'train_shape: {train_df.shape}\n vali_shape: {validation_test_df.shape}')

    print('sorting by index and step...')
    # sort the dataframes
    train_df.sort_values(['index', 'dummy_step'], inplace=True)
    train_df.drop('dummy_step', axis=1, inplace=True
                  )
    validation_test_df.sort_values(['index', 'dummy_step'], inplace=True)
    validation_test_df.drop('dummy_step', axis=1, inplace=True)

    print('after join')
    return train_df, validation_test_df, categorical_idxs

def create_lightGBM_dataset(mode, cluster, features_array, dataset_name):
    def _create_groups(df):
        """
        function used to retrieve the len of the groups
        :param df:
        :return:
        """
        df = df[['user_id', 'session_id']]
        group = df.groupby(['user_id', 'session_id'],
                           sort=False).apply(lambda x: len(x)).values
        return group

    def _save_dataset(base_path, mode, df):
        assert mode in ['train', 'vali'], 'the mode has to be train or vali'

        check_folder(base_path)

        x = df.drop(['index', 'user_id', 'session_id', 'item_id', 'label'], axis=1)
        x.to_csv(f'{_BASE_PATH}/x_{mode}.csv')
        print(f'x_{mode} saved at: {_BASE_PATH}/x_{mode}.csv')

        y = df['label'].values
        np.save(f'{_BASE_PATH}/y_{mode}', y)
        print(f'y_{mode} saved at: {_BASE_PATH}/y_{mode}.npy')

        groups = _create_groups(df)
        np.save(f'{_BASE_PATH}/groups_{mode}', groups)
        print(f'groups_{mode} saved at: {_BASE_PATH}/groups_{mode}.npy')

    # base save path
    _BASE_PATH = f'dataset/preprocessed/lightGBM/{cluster}/{mode}/{dataset_name}'

    # retrieve the TRAIN and VALIDATION/TEST data
    train_df, validation_df = merge_features(mode, cluster, features_array)

    _save_dataset(_BASE_PATH, 'train', train_df)
    _save_dataset(_BASE_PATH, 'vali', validation_df)


if __name__ == '__main__':
    features_array = [ImpressionRatingNumeric, ActionsInvolvingImpressionSession,
                      ImpressionLabel, ImpressionPriceInfoSession,
                      TimingFromLastInteractionImpression, TimesUserInteractedWithImpression,
                      ImpressionPositionSession, LastInteractionInvolvingImpression,
                      SessionDevice, SessionSortOrderWhenClickout, MeanPriceClickout,
                      PricePositionInfoInteractedReferences, SessionLength, TimeFromLastActionBeforeClk,
                      TimesImpressionAppearedInClickoutsSession]

    create_lightGBM_dataset('small', 'no_cluster', features_array, 'prova')
