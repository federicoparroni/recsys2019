import numpy as np

from extract_features.avg_price_interactions import AvgPriceInteractions
from extract_features.day_moment_in_day import DayOfWeekAndMomentInDay
from extract_features.impression_price_info_session_old import ImpressionPriceInfoSessionOld
from extract_features.last_action_involving_impression import LastActionInvolvingImpression
from extract_features.last_clickout_filters_satisfaction import LastClickoutFiltersSatisfaction
from extract_features.lazy_user import LazyUser
from extract_features.personalized_top_pop import PersonalizedTopPop
from extract_features.platform_features_similarity import PlatformFeaturesSimilarity
from extract_features.price_quality import PriceQuality
from extract_features.session_length_old import SessionLengthOld
from extract_features.session_num_clickouts import SessionNumClickouts
from extract_features.user_2_item_old import User2ItemOld
from preprocess_utils.merge_features import merge_features
from extract_features.impression_rating_numeric import ImpressionRatingNumeric
from extract_features.actions_involving_impression_session import ActionsInvolvingImpressionSession
from extract_features.frenzy_factor_consecutive_steps import FrenzyFactorSession
from extract_features.impression_position_session import ImpressionPositionSession
from extract_features.label import ImpressionLabel
from extract_features.mean_price_clickout import MeanPriceClickout
from extract_features.session_device import SessionDevice
from extract_features.session_sort_order_when_clickout import SessionSortOrderWhenClickout
from extract_features.times_impression_appeared_in_clickouts_session import TimesImpressionAppearedInClickoutsSession
from extract_features.times_user_interacted_with_impression import TimesUserInteractedWithImpression
from extract_features.timing_from_last_interaction_impression import TimingFromLastInteractionImpression
from extract_features.time_per_impression import TimePerImpression
from extract_features.change_impression_order_position_in_session import ChangeImpressionOrderPositionInSession
from extract_features.session_actions_num_ref_diff_from_impressions import SessionActionNumRefDiffFromImpressions
from extract_features.top_pop_per_impression import TopPopPerImpression
from extract_features.top_pop_interaction_clickout_per_impression import TopPopInteractionClickoutPerImpression
from extract_features.classifier.last_action_before_clickout import LastActionBeforeClickout
from extract_features.impression_stars_numeric import ImpressionStarsNumeric
from extract_features.last_steps_before_clickout import StepsBeforeLastClickout
from extract_features.location_reference_percentage_of_clickouts import LocationReferencePercentageOfClickouts
from extract_features.location_reference_percentage_of_interactions import LocationReferencePercentageOfInteractions
from extract_features.num_impressions_in_clickout import NumImpressionsInClickout
from extract_features.num_times_item_impressed import NumTimesItemImpressed
from extract_features.past_future_session_features import PastFutureSessionFeatures
from extract_features.perc_click_per_impressions import PercClickPerImpressions
from extract_features.platform_reference_percentage_of_clickouts import PlatformReferencePercentageOfClickouts
from extract_features.platform_reference_percentage_of_interactions import PlatformReferencePercentageOfInteractions
from extract_features.platform_session import PlatformSession

from pathlib import Path
from utils.check_folder import check_folder


def to_pool_dataset(dataset, save_dataset=True, path=''):
    print('Creating query-based dataset...')
    dataset['id'] = dataset.groupby(['user_id', 'session_id']).ngroup()
    groups = dataset.id.values
    print(groups)

    if save_dataset:
        np.save(path, groups)

    print('Query-based dataset created.')


def create_dataset(mode, cluster):
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

    curr_dir = Path(__file__).absolute().parent
    data_dir = curr_dir.joinpath('..', 'dataset/preprocessed/{}/{}/catboost/'.format(cluster, mode))
    print(data_dir)
    check_folder(str(data_dir))

    train_df, test_df, _, __ = merge_features(mode, cluster, features_array, merge_kind='left', onehot=False, create_not_existing_features=True)

    train_df = train_df.fillna(-1)
    test_df = test_df.fillna(-1)


    train_df.to_csv(str(data_dir) + '/train.csv', index=False)
    #to_pool_dataset(train_df, path=str(data_dir) + '/catboost_train.txt')

    print('Train saved')
    test_df.to_csv(str(data_dir) + '/test.csv', index=False)
    #to_pool_dataset(test_df, path=str(data_dir) + '/catboost_test.txt')


if __name__ == "__main__":
    from utils.menu import mode_selection, cluster_selection
    mode = mode_selection()
    cluster = cluster_selection()
    create_dataset(mode=mode, cluster=cluster)
