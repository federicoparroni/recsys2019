import lightgbm as lgb
import data
from tqdm import tqdm
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
from preprocess_utils.last_clickout_indices import expand_impressions
from utils.reduce_memory_usage_df import reduce_mem_usage
import multiprocessing

from scipy.sparse import save_npz
import data
import numpy as np
from tqdm import tqdm
import pandas as pd
from utils.check_folder import check_folder
import pickle
import utils.telegram_bot as Hera
from extract_features.actions_involving_impression_session import ActionsInvolvingImpressionSession
from extract_features.adjusted_location_reference_percentage_of_clickouts import AdjustedLocationReferencePercentageOfClickouts
from extract_features.adjusted_location_reference_percentage_of_interactions import AdjustedLocationReferencePercentageOfInteractions
from extract_features.adjusted_perc_click_per_impressions import AdjustedPercClickPerImpressions
from extract_features.platform_features_similarity import PlatformFeaturesSimilarity
from extract_features.adjusted_platform_reference_percentage_of_clickouts import AdjustedPlatformReferencePercentageOfClickouts
from extract_features.adjusted_platform_reference_percentage_of_interactions import AdjustedPlatformReferencePercentageOfInteractions
from extract_features.avg_price_interactions import AvgPriceInteractions
from extract_features.change_impression_order_position_in_session import ChangeImpressionOrderPositionInSession
from extract_features.changes_of_sort_order_before_current import ChangeOfSortOrderBeforeCurrent
from extract_features.city_session import CitySession
from extract_features.city_session_populars_only import CitySessionPopularsOnly
from extract_features.classifier_parro import ClassifierParro
from extract_features.classifier_piccio import ClassifierPiccio
from extract_features.country_searched_session import CountrySearchedSession
from extract_features.day_moment_in_day import DayOfWeekAndMomentInDay
from extract_features.fraction_pos_price import FractionPosPrice
from extract_features.frenzy_factor_consecutive_steps import FrenzyFactorSession
from extract_features.impression_features import ImpressionFeature
from extract_features.impression_position_in_percentage import ImpressionPositionInPercentage
from extract_features.impression_position_session import ImpressionPositionSession
from extract_features.impression_price_info_session import ImpressionPriceInfoSession
from extract_features.impression_rating import ImpressionRating
from extract_features.impression_rating_numeric import ImpressionRatingNumeric
from extract_features.impression_stars_numeric import ImpressionStarsNumeric
from extract_features.label import ImpressionLabel
#from extract_features.last_action_involving_impression import LastInteractionInvolvingImpression
from extract_features.last_clickout_filters_satisfaction import LastClickoutFiltersSatisfaction
from extract_features.last_steps_before_clickout import StepsBeforeLastClickout
from extract_features.lazy_user import LazyUser
from extract_features.location_features_similarity import LocationFeaturesSimilarity
from extract_features.location_reference_percentage_of_clickouts import LocationReferencePercentageOfClickouts
from extract_features.location_reference_percentage_of_interactions import LocationReferencePercentageOfInteractions
from extract_features.mean_price_clickout import MeanPriceClickout
from extract_features.normalized_platform_features_similarity import NormalizedPlatformFeaturesSimilarity
from extract_features.num_impressions_in_clickout import NumImpressionsInClickout
from extract_features.num_times_item_impressed import NumTimesItemImpressed
#from extract_features.past_future_session_features import PastFutureSessionFeatures
from extract_features.perc_click_per_impressions import PercClickPerImpressions
from extract_features.perc_click_per_pos import PercClickPerPos
from extract_features.personalized_top_pop import PersonalizedTopPop
#from extract_features.platform_features_similarty import PlatformFeaturesSimilarity
from extract_features.platform_reference_percentage_of_clickouts import PlatformReferencePercentageOfClickouts
from extract_features.platform_reference_percentage_of_interactions import PlatformReferencePercentageOfInteractions
from extract_features.platform_session import PlatformSession
#from extract_features.price_info_session import PriceInfoSession
from extract_features.price_quality import PriceQuality
from extract_features.ref_pop_after_first_position import RefPopAfterFirstPosition
from extract_features.session_actions_num_ref_diff_from_impressions import SessionActionNumRefDiffFromImpressions
from extract_features.session_device import SessionDevice
from extract_features.session_filters_active_when_clickout import SessionFilterActiveWhenClickout
from extract_features.session_length import SessionLength
from extract_features.session_num_clickouts import SessionNumClickouts
from extract_features.session_num_filter_sel import SessionNumFilterSel
from extract_features.session_num_inter_item_image import SessionNumInterItemImage
from extract_features.session_num_not_numeric import SessionNumNotNumeric
from extract_features.session_sort_order_when_clickout import SessionSortOrderWhenClickout
#from extract_features.time_from_last_action_before_clk import TimeFromLastActionBeforeClk
from extract_features.statistics_pos_interacted import StatisticsPosInteracted
from extract_features.statistics_time_from_last_action import StatisticsTimeFromLastAction
from extract_features.time_per_impression import TimePerImpression
from extract_features.times_impression_appeared_in_clickouts_session import TimesImpressionAppearedInClickoutsSession
from extract_features.times_user_interacted_with_impression import TimesUserInteractedWithImpression
from extract_features.timing_from_last_interaction_impression import TimingFromLastInteractionImpression
from extract_features.top_pop_interaction_clickout_per_impression import TopPopInteractionClickoutPerImpression
from extract_features.top_pop_interaction_sorting_filters import TopPopInteractionFilters
from extract_features.top_pop_per_impression import TopPopPerImpression
from extract_features.top_pop_sorting_filters import TopPopSortingFilters
from extract_features.user_2_item import User2Item
from extract_features.user_feature import UserFeature

import utils.menu as menu
from time import time

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
    time_joins = 0
    for f in features_array:
        _feature = f(mode=mode, cluster='no_cluster')
        feature = _feature.read_feature(one_hot=False)

        print(f'shape of feature: {feature.shape}\n')
        print(f'len of feature:{len(feature)}\n')

        start = time()
        train_df = train_df.merge(feature)
        validation_test_df = validation_test_df.merge(feature)
        print(f'time to do the join: {time()-start}')
        time_joins += time()-start
        print(f'train_shape: {train_df.shape}\n vali_shape: {validation_test_df.shape}')

    print(f'total time to do joins: {time_joins}')

    print('sorting by index and step...')
    # sort the dataframes
    train_df.sort_values(['index', 'dummy_step'], inplace=True)
    train_df.drop('dummy_step', axis=1, inplace=True
                  )
    validation_test_df.sort_values(['index', 'dummy_step'], inplace=True)
    validation_test_df.drop('dummy_step', axis=1, inplace=True)

    print('after join')
    return train_df, validation_test_df

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
        print('reducing memory usage...')
        df = reduce_mem_usage(df)

        check_folder(base_path)

        x = df.drop(['index', 'user_id', 'session_id', 'item_id', 'label'], axis=1)
        x.to_hdf(f'{_BASE_PATH}/x_{mode}.hdf', key='df', index=False, format='table')
        print(f'x_{mode} saved at: {_BASE_PATH}/x_{mode}.hdf')

        y = df['label'].values
        np.save(f'{_BASE_PATH}/y_{mode}', y)
        print(f'y_{mode} saved at: {_BASE_PATH}/y_{mode}.npy')

        groups = _create_groups(df)
        np.save(f'{_BASE_PATH}/groups_{mode}', groups)
        print(f'groups_{mode} saved at: {_BASE_PATH}/groups_{mode}.npy')

        user_session_item = df[['user_id', 'session_id', 'item_id']]
        user_session_item.to_csv(f'{_BASE_PATH}/user_session_item_{mode}.csv', index=False)
        print(f'user_session_item_{mode} saved at: {_BASE_PATH}/user_session_item_{mode}.csv')

    # base save path
    _BASE_PATH = f'dataset/preprocessed/lightGBM/{cluster}/{mode}/{dataset_name}'

    # retrieve the TRAIN and VALIDATION/TEST data
    train_df, validation_df = merge_features_lgb(mode, cluster, features_array)

    print('saving features names...')
    check_folder(f"{_BASE_PATH}")
    with open(f"{_BASE_PATH}/Features.txt", "w+") as text_file:
        text_file.write(str([str(fn) for fn in features_array]))

    Hera.send_message('SAVING TRAIN LIGHTGBM...')
    _save_dataset(_BASE_PATH, 'train', train_df)
    Hera.send_message('SAVING VALI LIGHTGBM...')
    _save_dataset(_BASE_PATH, 'vali', validation_df)
    Hera.send_message('PROCEDURE ENDED CORRECTLY')


if __name__ == '__main__':

    features_array = [
        ImpressionLabel,
        StatisticsPosInteracted,
        # AdjustedLocationReferencePercentageOfClickouts,
        # AdjustedLocationReferencePercentageOfInteractions,
        # AdjustedPercClickPerImpressions,
        PlatformFeaturesSimilarity,
        # AdjustedPlatformReferencePercentageOfClickouts,
        # AdjustedPlatformReferencePercentageOfInteractions,
        AvgPriceInteractions,
        ChangeImpressionOrderPositionInSession,
        #CountrySearchedSession,
        DayOfWeekAndMomentInDay,
        FractionPosPrice,
        FrenzyFactorSession,
        ImpressionPositionInPercentage,
        ImpressionPositionSession,
        ImpressionPriceInfoSession,
        ImpressionRatingNumeric,
        ImpressionStarsNumeric,
        ##LastInteractionInvolvingImpression,
        LastClickoutFiltersSatisfaction,
        StepsBeforeLastClickout,
        LazyUser,
        LocationFeaturesSimilarity,
        LocationReferencePercentageOfClickouts,
        ##LocationReferencePercentageOfInteractions,
        MeanPriceClickout,
        NormalizedPlatformFeaturesSimilarity,
        NumImpressionsInClickout,
        NumTimesItemImpressed,
        PercClickPerImpressions,
        # PercClickPerPos,
        # PersonalizedTopPop,
        PlatformReferencePercentageOfClickouts,
        ##PlatformReferencePercentageOfInteractions,
        PriceQuality,
        # RefPopAfterFirstPosition,
        SessionActionNumRefDiffFromImpressions,
        SessionDevice,
        SessionFilterActiveWhenClickout,
        SessionLength,
        # SessionNumClickouts,
        # SessionNumFilterSel,
        # SessionNumInterItemImage,
        # SessionNumNotNumeric,
        SessionSortOrderWhenClickout,
        StatisticsTimeFromLastAction,
        TimePerImpression,
        TimesUserInteractedWithImpression,
        TimingFromLastInteractionImpression,
        TopPopInteractionClickoutPerImpression,
        TopPopPerImpression,
        User2Item,
        #UserFeature
    ]

    mode=single_choice('select mode:', ['full', 'local', 'small'])
    cluster=single_choice('select cluster:', ['no_cluster'])
    #dataset_name=single_choice('select dataset name:',['prova', 'dataset1', 'dataset2', 'old'])
    dataset_name = input('insert the dataset name:\n')
    create_lightGBM_dataset(mode=mode, cluster=cluster, features_array=features_array,
                            dataset_name=dataset_name)
