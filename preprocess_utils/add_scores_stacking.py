from scipy.sparse import save_npz
import data
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from utils.check_folder import check_folder
from extract_features.actions_involving_impression_session import ActionsInvolvingImpressionSession
from extract_features.change_impression_order_position_in_session import ChangeImpressionOrderPositionInSession
from extract_features.changes_of_sort_order_before_current import ChangeOfSortOrderBeforeCurrent
from extract_features.city_session import CitySession
from extract_features.city_session_populars_only import CitySessionPopularsOnly
from extract_features.classifier_parro import ClassifierParro
from extract_features.classifier_piccio import ClassifierPiccio
from extract_features.country_searched_session import CountrySearchedSession
from extract_features.day_moment_in_day import DayOfWeekAndMomentInDay
from extract_features.frenzy_factor_consecutive_steps import FrenzyFactorSession
from extract_features.impression_features import ImpressionFeature
from extract_features.impression_position_session import ImpressionPositionSession
from extract_features.impression_price_info_session import ImpressionPriceInfoSession
from extract_features.impression_price_info_session_old import ImpressionPriceInfoSessionOld
from extract_features.impression_rating import ImpressionRating
from extract_features.impression_rating_numeric import ImpressionRatingNumeric
from extract_features.impression_stars_numeric import ImpressionStarsNumeric
from extract_features.impression_features_cleaned import ImpressionFeatureCleaned
from extract_features.label import ImpressionLabel
from extract_features.last_action_involving_impression import LastActionInvolvingImpression
from extract_features.last_clickout_filters_satisfaction import LastClickoutFiltersSatisfaction
from extract_features.last_steps_before_clickout import StepsBeforeLastClickout
from extract_features.lazy_user import LazyUser
from extract_features.location_reference_percentage_of_clickouts import LocationReferencePercentageOfClickouts
from extract_features.location_reference_percentage_of_interactions import LocationReferencePercentageOfInteractions
from extract_features.num_impressions_in_clickout import NumImpressionsInClickout
from extract_features.num_times_item_impressed import NumTimesItemImpressed
from extract_features.perc_click_per_impressions import PercClickPerImpressions
from extract_features.personalized_top_pop import PersonalizedTopPop
from extract_features.platform_features_similarity import PlatformFeaturesSimilarity
from extract_features.platform_reference_percentage_of_clickouts import PlatformReferencePercentageOfClickouts
from extract_features.platform_reference_percentage_of_interactions import PlatformReferencePercentageOfInteractions
from extract_features.platform_session import PlatformSession
from extract_features.price_info_session import PriceInfoSession
from extract_features.price_quality import PriceQuality
from extract_features.session_actions_num_ref_diff_from_impressions import SessionActionNumRefDiffFromImpressions
from extract_features.session_device import SessionDevice
from extract_features.session_filters_active_when_clickout import SessionFilterActiveWhenClickout
from extract_features.session_length import SessionLength
from extract_features.session_length_old import SessionLengthOld
from extract_features.session_sort_order_when_clickout import SessionSortOrderWhenClickout
from extract_features.session_num_clickouts import SessionNumClickouts
from extract_features.session_num_filter_sel import SessionNumFilterSel
from extract_features.session_num_inter_item_image import SessionNumInterItemImage
from extract_features.session_num_not_numeric import SessionNumNotNumeric
from extract_features.time_per_impression import TimePerImpression
from extract_features.times_impression_appeared_in_clickouts_session import TimesImpressionAppearedInClickoutsSession
from extract_features.timing_from_last_interaction_impression import TimingFromLastInteractionImpression
from extract_features.timing_from_last_interaction_impression_old import TimingFromLastInteractionImpressionOld
from extract_features.top_pop_interaction_clickout_per_impression import TopPopInteractionClickoutPerImpression
from extract_features.top_pop_per_impression import TopPopPerImpression
from extract_features.top_pop_sorting_filters import TopPopSortingFilters
from extract_features.user_2_item import User2Item
from extract_features.user_2_item_old import User2ItemOld
from extract_features.adjusted_location_reference_percentage_of_clickouts import AdjustedLocationReferencePercentageOfClickouts
from extract_features.adjusted_location_reference_percentage_of_interactions import AdjustedLocationReferencePercentageOfInteractions
from extract_features.adjusted_perc_click_per_impressions import AdjustedPercClickPerImpressions
from extract_features.adjusted_platform_reference_percentage_of_clickouts import AdjustedPlatformReferencePercentageOfClickouts
from extract_features.adjusted_platform_reference_percentage_of_interactions import AdjustedPlatformReferencePercentageOfInteractions
from extract_features.location_features_similarity import LocationFeaturesSimilarity
from extract_features.normalized_platform_features_similarity import NormalizedPlatformFeaturesSimilarity
from extract_features.ref_pop_after_first_position import RefPopAfterFirstPosition
from extract_features.user_feature import UserFeature
from extract_features.perc_click_per_pos import PercClickPerPos
from extract_features.mean_price_clickout import MeanPriceClickout
from extract_features.avg_price_interactions import AvgPriceInteractions
from extract_features.times_user_interacted_with_impression import TimesUserInteractedWithImpression
from extract_features.classifier.last_action_before_clickout import LastActionBeforeClickout
from extract_features.fraction_pos_price import FractionPosPrice
from extract_features.past_future_session_features import PastFutureSessionFeatures
from extract_features.normalized_platform_features_similarity import NormalizedPlatformFeaturesSimilarity
from utils.menu import single_choice
from preprocess_utils.merge_features import merge_features
from os.path import join
from shutil import copyfile


def create_groups(df):
    df = df[['user_id', 'session_id']]
    group = df.groupby(['user_id', 'session_id'],
                       sort=False).apply(lambda x: len(x)).values
    return group


def create_weights(df):
    df_slice = df[['user_id', 'session_id', 'impression_position', 'label']]
    weights = []
    au = df_slice.head().user_id.values[0]
    ai = df_slice.head().session_id.values[0]
    found = False
    for idx, row in df_slice.iterrows():
        if au != row.user_id or ai != row.session_id:
            if not found and len(weights) > 0:
                weights.append(1)
            au = row.user_id
            ai = row.session_id
            found = False
        if row.label == 1:
            if row.impression_position == 1:
                weights.append(0.5)
            else:
                weights.append(2)
            found = True
    return weights


def create_dataset(mode, cluster, class_weights=False):
    # training
    kind = input('insert the kind: ')

    scores_array = [
        'rnn_classifier.csv.gz',
        'rnn_no_bias_balanced.csv.gz',
        'scores_softmax_loss.csv.gz',
        'xgboost_impr_features.csv.gz',
        'rnn_GRU_2layers_64units_2dense_noclass0.csv.gz',
        'scores_pairwise_soft_zero_one_loss.csv.gz',
        'xgb_forte_700.csv.gz',
    ]

    path_hdf = 'dataset/preprocessed/{}/{}/xgboost/{}/base.hdf'.format(cluster, mode, 'base_dataset_stacking')
    train_df = pd.read_hdf(path_hdf, key = 'train')
    test_df = pd.read_hdf(path_hdf, key = 'test')

    if len(scores_array) > 0:
        for path in scores_array:
            score = pd.read_csv('scores/{}'.format(path))

            if 'item_id' in score.columns:
                print('item_id found')
                cols = [c for c in score.columns if c in ['user_id', 'session_id', 'item_id'] or 'score' in c]
                score = score[cols]
                score = score.groupby(['user_id', 'session_id', 'item_id'], as_index=False).last()
                train_df = train_df.merge(score, on=['user_id', 'session_id', 'item_id'], how='left')
                test_df = test_df.merge(score, on=['user_id', 'session_id', 'item_id'], how='left')
                print(f'train_shape: {train_df.shape}\n vali_shape: {test_df.shape}')
            
            else:
                print('item_id not found')
                cols = [c for c in score.columns if c in ['user_id', 'session_id'] or 'score' in c]
                score = score[cols]
                score = score.groupby(['user_id', 'session_id'], as_index=False).last()
                train_df = train_df.merge(score, on=['user_id', 'session_id'], how='left')
                test_df = test_df.merge(score, on=['user_id', 'session_id'], how='left')
                print(f'train_shape: {train_df.shape}\n vali_shape: {test_df.shape}')

    train_df = train_df.replace(-1, np.nan)
    test_df = test_df.replace(-1, np.nan)

    bp = 'dataset/preprocessed/{}/{}/xgboost/{}/'.format(cluster, mode, kind)
    check_folder(bp)

    if class_weights:
        weights = train_df[['user_id', 'session_id',
                            'weights']].drop_duplicates().weights.values
        print(len(weights))
        np.save(join(bp, 'class_weights'), weights)
        print('class weights saved')

    if class_weights:
        X_train = train_df.drop(
            ['index', 'user_id', 'session_id', 'item_id', 'label', 'weights'], axis=1)
    else:
        X_train = train_df.drop(
            ['index', 'user_id', 'session_id', 'item_id', 'label'], axis=1)
    print(','.join(X_train.columns.values))
    X_train = X_train.to_sparse(fill_value=0)
    X_train = X_train.astype(np.float64)
    X_train = X_train.to_coo().tocsr()
    save_npz(join(bp, 'X_train'), X_train)
    print('X_train saved')

    user_session_item = train_df[['user_id', 'session_id', 'item_id']]
    user_session_item.to_csv(join(bp, 'user_session_item_train.csv'), index=False)

    y_train = train_df[['label']]
    y_train.to_csv(join(bp, 'y_train.csv'))
    print('y_train saved')

    group = create_groups(train_df)
    print(len(group))
    np.save(join(bp, 'group_train'), group)
    print('train groups saved')

    # np.save(join(bp, 'train_indices'), train_idxs)
    copyfile('dataset/preprocessed/{}/{}/xgboost/{}/train_indices.npy'.format(cluster, mode, 'base_dataset_stacking'),
                join(bp, 'train_indices.npy'))

    print('train data completed')

    if class_weights:
        X_test = test_df.drop(
            ['index', 'user_id', 'session_id', 'item_id', 'label', 'weights'], axis=1)
    else:
        X_test = test_df.drop(
            ['index', 'user_id', 'session_id', 'item_id', 'label'], axis=1)

    X_test = X_test.to_sparse(fill_value=0)
    X_test = X_test.astype(np.float64)
    X_test = X_test.to_coo().tocsr()
    save_npz(join(bp, 'X_test'), X_test)
    print('X_test saved')

    user_session_item = test_df[['user_id', 'session_id', 'item_id']]
    user_session_item.to_csv(join(bp, 'user_session_item_test.csv'), index=False)

    y_test = test_df[['label']]
    y_test.to_csv(join(bp, 'y_test.csv'))
    print('y_test saved')

    group = create_groups(test_df)
    print(len(group))
    np.save(join(bp, 'group_test'), group)

    print('test groups saved')

    print('test data completed')


if __name__ == "__main__":
    from utils.menu import mode_selection
    from utils.menu import cluster_selection
    mode = mode_selection()
    cluster = cluster_selection()
    create_dataset(mode, cluster)
