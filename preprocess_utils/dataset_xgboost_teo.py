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
from utils.menu import single_choice
from preprocess_utils.merge_features import merge_features
from os.path import join
from extract_features.past_future_session_features import PastFutureSessionFeatures
from extract_features.normalized_platform_features_similarity import NormalizedPlatformFeaturesSimilarity
import math

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

def create_weights_position(train_df, mode,cluster):
    train = data.train_df(mode, cluster)
    test = data.test_df(mode, cluster)
    df = pd.concat([train, test])
    # get for each user-session the position of the clicked item
    df_clks = df[(df['reference'].str.isnumeric()==True)&(df['action_type']=='clickout item')][['user_id','session_id','reference','impressions']]
    df_clks.impressions = df_clks.impressions.str.split('|')
    new_col = []
    for t in tqdm(zip(df_clks.reference, df_clks.impressions)):
        if t[0] in t[1]:
            new_col.append(t[1].index(t[0])+1)
        else:
            new_col.append(-1)
    df_clks['pos_clicked'] = new_col
    pos_clicked_list = df_clks.pos_clicked.tolist()
    # create dictionary {pos:score}
    dict_pos_score = {}
    for i in tqdm(range(1,26)):
        dict_pos_score[i] = 1-(pos_clicked_list.count(i)/len(pos_clicked_list)) # the function is 1-(#pos/tot_rowså)
    # group per user-session
    group = train_df.drop_duplicates(['user_id','session_id'])[['user_id','session_id']].reset_index(drop=True)
    # assign weight
    gr = train_df[train_df.label==1][['user_id','session_id','impression_position']]
    new_col = []
    for p in gr.impression_position:
        if p not in range(1,26):
            new_col.append(0)
        else:
            new_col.append(dict_pos_score[p])
    gr['weight'] = new_col
    final = pd.merge(group, gr, how='left', on=['user_id','session_id']).fillna(0)
    sample_weights = final['weight'].values
    return sample_weights

def create_log_weights(train_df):
    d = {}
    for i in range(1,26):
        d[i]=math.sqrt(math.log(1+i, 26))
    # group per user-session
    group = train_df.drop_duplicates(['user_id','session_id'])[['user_id','session_id']].reset_index(drop=True)
    # assign weight
    gr = train_df[train_df.label==1][['user_id','session_id','impression_position']]
    new_col = []
    for p in gr.impression_position:
        if p not in range(1,26):
            new_col.append(0)
        else:
            new_col.append(d[p])
    gr['weight'] = new_col
    final = pd.merge(group, gr, how='left', on=['user_id','session_id']).fillna(0)
    sample_weights = final['weight'].values
    return sample_weights


def create_dataset(mode, cluster, class_weights=False, weights_position=True, log_weights=True):
    # training
    kind = input('insert the kind: ')
    if cluster == 'no_cluster' or True:

        if kind == 'kind2':
            # questo fa 0.6755 in locale + NormalizedPlatformFeaturesSimilarity, SessionNumClickouts fa 0.67588
                features_array = [
                ImpressionLabel,
                PastFutureSessionFeatures
                ]
        if kind == 'kind3':
            # questo è quello che usa dani su cat
            features_array = [
                ActionsInvolvingImpressionSession,
                ImpressionPositionSession,
                ImpressionPriceInfoSessionOld,
                ImpressionRatingNumeric,
                ImpressionLabel,
                LastActionInvolvingImpression,
                MeanPriceClickout,
                AvgPriceInteractions,
                SessionDevice,
                NumImpressionsInClickout,
                SessionLengthOld,
                TimesImpressionAppearedInClickoutsSession,
                TimesUserInteractedWithImpression,
                TimingFromLastInteractionImpression,
                TopPopPerImpression,
                TopPopInteractionClickoutPerImpression,
                ChangeImpressionOrderPositionInSession,
                FrenzyFactorSession,
                DayOfWeekAndMomentInDay,
                LastClickoutFiltersSatisfaction,
                TimePerImpression,
                PersonalizedTopPop,
                PriceQuality,
                PlatformFeaturesSimilarity,
                LastActionBeforeClickout,
                ImpressionStarsNumeric,
                StepsBeforeLastClickout,
                LocationReferencePercentageOfClickouts,
                LocationReferencePercentageOfInteractions,
                NumTimesItemImpressed,
                PercClickPerImpressions,
                PlatformReferencePercentageOfClickouts,
                PlatformReferencePercentageOfInteractions,
                PlatformSession,
                User2ItemOld,
                LazyUser,
                PastFutureSessionFeatures
            ]
        if kind == 'kind1':
            # questo fa 0.6755 in locale coi param magici e senza NormalizedPlatformFeaturesSimilarity e SessionNumClickouts
            # fa 0.67566 con i seguenti params:
            # learning_rate=0.1366 min_child_weight=1 n_estimators=499
            # max_depth=10 subsample=1 colsample_bytree=1 reg_lambda=4.22 reg_alpha=10.72
            # fa 0.67588 con anche NormalizedPlatformFeaturesSimilarity e SessionNumClickouts
            features_array = [
                ImpressionPositionSession,
                ImpressionPriceInfoSessionOld,
                ImpressionRatingNumeric,
                ImpressionLabel,
                LastActionInvolvingImpression,
                MeanPriceClickout,
                AvgPriceInteractions,
                SessionDevice,
                NumImpressionsInClickout,
                SessionLengthOld,
                TimesImpressionAppearedInClickoutsSession,
                TimesUserInteractedWithImpression,
                TimingFromLastInteractionImpression,
                TopPopPerImpression,
                TopPopInteractionClickoutPerImpression,
                ChangeImpressionOrderPositionInSession,
                FrenzyFactorSession,
                DayOfWeekAndMomentInDay,
                LastClickoutFiltersSatisfaction,
                TimePerImpression,
                PersonalizedTopPop,
                PriceQuality,
                PlatformFeaturesSimilarity,
                LastActionBeforeClickout,
                ImpressionStarsNumeric,
                StepsBeforeLastClickout,
                LocationReferencePercentageOfClickouts,
                LocationReferencePercentageOfInteractions,
                NumTimesItemImpressed,
                PercClickPerImpressions,
                PlatformReferencePercentageOfClickouts,
                PlatformReferencePercentageOfInteractions,
                PlatformSession,
                User2ItemOld,
                LazyUser,
                PastFutureSessionFeatures,
                SessionSortOrderWhenClickout,
                SessionActionNumRefDiffFromImpressions,
                ActionsInvolvingImpressionSession,
                SessionNumClickouts
            ]

    train_df, test_df, train_idxs, _ = merge_features(mode, cluster, features_array, merge_kind='left')

    train_df = train_df.replace(-1, np.nan)
    test_df = test_df.replace(-1, np.nan)

    bp = 'dataset/preprocessed/{}/{}/xgboost/{}/'.format(cluster, mode, kind)
    check_folder(bp)
    train_df.to_csv(join(bp, 'train_df.csv'))

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

    if weights_position:
        weights = create_weights_position(train_df, mode,cluster)
        print(len(weights))
        np.save(join(bp, 'weights_position'), weights)
        print('weights_position saved')

    if log_weights:
        lg_w = create_log_weights(train_df)
        print(len(lg_w))
        np.save(join(bp, 'log_weights'), lg_w)
        print('log_weights saved')

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

    np.save(join(bp, 'train_indices'), train_idxs)

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
