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
from extract_features.impression_rating import ImpressionRating
from extract_features.impression_rating_numeric import ImpressionRatingNumeric
from extract_features.impression_stars_numeric import ImpressionStarsNumeric
from extract_features.label import ImpressionLabel
from extract_features.last_action_involving_impression import LastInteractionInvolvingImpression
from extract_features.last_clickout_filters_satisfaction import LastClickoutFiltersSatisfaction
from extract_features.last_steps_before_clickout import StepsBeforeLastClickout
from extract_features.lazy_user import LazyUser
from extract_features.location_reference_percentage_of_clickouts import LocationReferencePercentageOfClickouts
from extract_features.location_reference_percentage_of_interactions import LocationReferencePercentageOfInteractions
from extract_features.num_impressions_in_clickout import NumImpressionsInClickout
from extract_features.num_times_item_impressed import NumTimesItemImpressed
from extract_features.perc_click_per_impressions import PercClickPerImpressions
from extract_features.personalized_top_pop import PersonalizedTopPop
from extract_features.platform_features_similarty import PlatformFeaturesSimilarity
from extract_features.platform_reference_percentage_of_clickouts import PlatformReferencePercentageOfClickouts
from extract_features.platform_reference_percentage_of_interactions import PlatformReferencePercentageOfInteractions
from extract_features.platform_session import PlatformSession
from extract_features.price_info_session import PriceInfoSession
from extract_features.price_position_info_interactions import PricePositionInfoInteractedReferences
from extract_features.price_quality import PriceQuality
from extract_features.session_actions_num_ref_diff_from_impressions import SessionActionNumRefDiffFromImpressions
from extract_features.session_device import SessionDevice
from extract_features.session_filters_active_when_clickout import SessionFilterActiveWhenClickout
from extract_features.session_length import SessionLength
from extract_features.session_sort_order_when_clickout import SessionSortOrderWhenClickout
from extract_features.time_from_last_action_before_clk import TimeFromLastActionBeforeClk
from extract_features.time_per_impression import TimePerImpression
from extract_features.times_impression_appeared_in_clickouts_session import TimesImpressionAppearedInClickoutsSession
from extract_features.timing_from_last_interaction_impression import TimingFromLastInteractionImpression
from extract_features.top_pop_interaction_clickout_per_impression import TopPopInteractionClickoutPerImpression
from extract_features.top_pop_per_impression import TopPopPerImpression
from extract_features.top_pop_sorting_filters import TopPopSortingFilters
from extract_features.user_2_item import User2Item
from utils.menu import single_choice
from preprocess_utils.merge_features import merge_features
from os.path import join


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
    kind = single_choice(['1', '2', '3'], ['kind1', 'kind2', 'kind3'])
    if cluster == 'no_cluster':
        if kind == 'kind1':
            features_array = [PriceQuality, PlatformFeaturesSimilarity, ClassifierParro,
                            ClassifierPiccio, PersonalizedTopPop, TimePerImpression,
                            DayOfWeekAndMomentInDay, LastClickoutFiltersSatisfaction,
                            FrenzyFactorSession, ChangeImpressionOrderPositionInSession, 
                            User2Item, PlatformSession, PlatformReferencePercentageOfInteractions, 
                            PercClickPerImpressions, PlatformReferencePercentageOfClickouts,
                            NumImpressionsInClickout, NumTimesItemImpressed,
                            LocationReferencePercentageOfClickouts, LocationReferencePercentageOfInteractions,
                            StepsBeforeLastClickout, ImpressionStarsNumeric,
                            TopPopPerImpression, TopPopInteractionClickoutPerImpression, 
                            ImpressionRatingNumeric, ActionsInvolvingImpressionSession,
                            ImpressionLabel, ImpressionPriceInfoSession,
                            TimingFromLastInteractionImpression,
                            ImpressionPositionSession, LastInteractionInvolvingImpression,
                            SessionDevice, SessionSortOrderWhenClickout,
                            PricePositionInfoInteractedReferences, SessionLength, TimeFromLastActionBeforeClk,
                            TimesImpressionAppearedInClickoutsSession]
        elif kind == 'kind2':
            features_array = [PriceQuality, PlatformFeaturesSimilarity, PersonalizedTopPop, TimePerImpression,
                            DayOfWeekAndMomentInDay, LastClickoutFiltersSatisfaction,
                            FrenzyFactorSession, ChangeImpressionOrderPositionInSession, 
                            User2Item, PlatformSession, PlatformReferencePercentageOfInteractions, 
                            PercClickPerImpressions, PlatformReferencePercentageOfClickouts,
                            NumImpressionsInClickout, NumTimesItemImpressed,
                            LocationReferencePercentageOfClickouts, LocationReferencePercentageOfInteractions,
                            StepsBeforeLastClickout, ImpressionStarsNumeric,
                            TopPopPerImpression, TopPopInteractionClickoutPerImpression, 
                            ImpressionRatingNumeric, ActionsInvolvingImpressionSession,
                            ImpressionLabel, ImpressionPriceInfoSession,
                            TimingFromLastInteractionImpression,
                            ImpressionPositionSession, LastInteractionInvolvingImpression,
                            SessionDevice, SessionSortOrderWhenClickout,
                            PricePositionInfoInteractedReferences, SessionLength, TimeFromLastActionBeforeClk,
                            TimesImpressionAppearedInClickoutsSession]
        elif kind == 'kind3':
            features_array = [
            #ActionsInvolvingImpressionSession,
            # ChangeImpressionOrderPositionInSession,
            # ChangeOfSortOrderBeforeCurrent,
            # CitySession,
            CitySessionPopularsOnly,
            # ClassifierParro,
            # ClassifierPiccio,
            # CountrySearchedSession,
            # DayOfWeekAndMomentInDay,
            FrenzyFactorSession,
            # ImpressionFeature,
            ImpressionPositionSession,
            ImpressionPriceInfoSession,
            # ImpressionRating,
            ImpressionRatingNumeric,
            ImpressionStarsNumeric,
            ImpressionLabel,
            # LastInteractionInvolvingImpression,
            # LastClickoutFiltersSatisfaction,
            # StepsBeforeLastClickout,
            LazyUser,
            LocationReferencePercentageOfClickouts,
            LocationReferencePercentageOfInteractions,
            NumImpressionsInClickout,
            NumTimesItemImpressed,
            PercClickPerImpressions,
            PersonalizedTopPop,
            # PlatformFeaturesSimilarity,
            PlatformReferencePercentageOfClickouts,
            PlatformReferencePercentageOfInteractions,
            PlatformSession,
            PriceInfoSession,
            PriceQuality,
            SessionActionNumRefDiffFromImpressions,
            SessionDevice,
            SessionFilterActiveWhenClickout,
            SessionLength,
            # SessionSortOrderWhenClickout,
            # TimeFromLastActionBeforeClk,
            TimePerImpression,
            TimesImpressionAppearedInClickoutsSession,
            # TimingFromLastInteractionImpression,
            TopPopInteractionClickoutPerImpression,
            TopPopPerImpression,
            # TopPopSortingFilters,
            User2Item,
        ]

    if cluster == 'no_numeric_reference_no_one_step':
        features_array = [ChangeImpressionOrderPositionInSession, ChangeOfSortOrderBeforeCurrent, ImpressionLabel, 
                          DayOfWeekAndMomentInDay, FrenzyFactorSession, ImpressionPositionSession, 
                          ImpressionPriceInfoSession, ImpressionRatingNumeric, ImpressionStarsNumeric,
                          LastClickoutFiltersSatisfaction, StepsBeforeLastClickout, LocationReferencePercentageOfClickouts,
                          LocationReferencePercentageOfInteractions, NumImpressionsInClickout,
                          PercClickPerImpressions, PersonalizedTopPop, PlatformReferencePercentageOfClickouts,
                          PlatformReferencePercentageOfInteractions, PlatformSession, SessionDevice, 
                          SessionLength, TimeFromLastActionBeforeClk, TopPopInteractionClickoutPerImpression, 
                          TopPopPerImpression]

    train_df, test_df, train_idxs, _ = merge_features(mode, cluster, features_array, merge_kind='left')

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
    X_train = X_train.to_sparse(fill_value=0)
    X_train = X_train.astype(np.float64)
    X_train = X_train.to_coo().tocsr()
    save_npz(join(bp, 'X_train'), X_train)
    print('X_train saved')

    y_train = train_df[['label']]
    y_train.to_csv(join(bp, 'y_train.csv'))
    print('y_train saved')

    group = create_groups(train_df)
    print(len(group))
    np.save(join(bp, 'group_train'), group)
    print('train groups saved')

    np.save(join(bp, 'train_indices'), train_idxs)

    print('train data completed')

    if class_weights:
        X_test = test_df.drop(
            ['index', 'user_id', 'session_id', 'item_id', 'label', 'weights'], axis=1)
    else:
        X_test = test_df.drop(
            ['index', 'user_id', 'session_id', 'item_id', 'label'], axis=1)

    #if mode == 'full':
    X_test = X_test.to_sparse(fill_value=0)
    X_test = X_test.astype(np.float64)
    X_test = X_test.to_coo().tocsr()
    save_npz(join(bp, 'X_test'), X_test)
    #else:
    #    X_test.to_csv(join(bp, 'X_test.csv'))
    print('X_test saved')

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
