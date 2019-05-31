import numpy as np
from extract_features.city_session import CitySession
from extract_features.city_session_populars_only import CitySessionPopularsOnly
from extract_features.country_searched_session import CountrySearchedSession
from extract_features.day_moment_in_day import DayOfWeekAndMomentInDay
from extract_features.last_clickout_filters_satisfaction import LastClickoutFiltersSatisfaction
from extract_features.platform_features_similarty import PlatformFeaturesSimilarity
from extract_features.user_2_item import User2Item
from preprocess_utils.merge_features import merge_features
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
from extract_features.change_impression_order_position_in_session import ChangeImpressionOrderPositionInSession
from extract_features.session_actions_num_ref_diff_from_impressions import SessionActionNumRefDiffFromImpressions
from extract_features.top_pop_per_impression import TopPopPerImpression
from extract_features.top_pop_interaction_clickout_per_impression import TopPopInteractionClickoutPerImpression
from extract_features.classifier_piccio import ClassifierPiccio
from extract_features.classifier_parro import ClassifierParro
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
import os
from pathlib import Path


def to_pool_dataset(dataset, save_dataset=True, path=''):
    print('Creating query-based dataset...')
    dataset['id'] = dataset.groupby(['user_id', 'session_id']).ngroup()
    groups = dataset.id.values
    print(groups)

    if save_dataset:
        np.save(path, groups)

    print('Query-based dataset created.')


def create_dataset(mode, cluster):
    # training
    features_array = [ClassifierParro, ClassifierPiccio, PlatformFeaturesSimilarity, PastFutureSessionFeatures,
                      SessionFilterActiveWhenClickout, SessionActionNumRefDiffFromImpressions, DayOfWeekAndMomentInDay,
                      LastClickoutFiltersSatisfaction, CitySession, CountrySearchedSession,
                      FrenzyFactorSession, ChangeImpressionOrderPositionInSession,
                      User2Item, PlatformSession, PlatformReferencePercentageOfInteractions,
                      PercClickPerImpressions, PlatformReferencePercentageOfClickouts,
                      NumImpressionsInClickout, NumTimesItemImpressed,
                      LocationReferencePercentageOfClickouts, LocationReferencePercentageOfInteractions,
                      StepsBeforeLastClickout, ImpressionStarsNumeric, LastActionBeforeClickout,
                      TopPopPerImpression, TopPopInteractionClickoutPerImpression,
                      ImpressionRatingNumeric, ActionsInvolvingImpressionSession,
                      ImpressionLabel, ImpressionPriceInfoSession,
                      TimingFromLastInteractionImpression, TimesUserInteractedWithImpression,
                      ImpressionPositionSession, LastInteractionInvolvingImpression,
                      SessionDevice, SessionSortOrderWhenClickout, MeanPriceClickout,
                      PricePositionInfoInteractedReferences, SessionLength, TimeFromLastActionBeforeClk,
                      TimesImpressionAppearedInClickoutsSession]

    curr_dir = Path(__file__).absolute().parent
    data_dir = curr_dir.joinpath('..', 'dataset/preprocessed/{}/{}/catboost/'.format(cluster, mode))
    print(data_dir)
    check_folder(str(data_dir))

    train_df, test_df = merge_features(mode, cluster, features_array, onehot=False)

    if os.path.isfile(str(data_dir) + '/catboost_train.txt'):
        print('Train File già presente')
    else:
        train_df.to_csv(str(data_dir) + '/train.csv', index=False)
        to_pool_dataset(train_df, path=str(data_dir) + '/catboost_train.txt')

    if os.path.isfile(str(data_dir) + '/test.csv'):
        print('Test File già presente')
    else:
        test_df.to_csv(str(data_dir) + '/test.csv', index=False)
        to_pool_dataset(test_df, path=str(data_dir) + '/catboost_test.txt')


if __name__ == "__main__":
    from utils.menu import mode_selection
    mode = mode_selection()
    create_dataset(mode='small', cluster='no_cluster')
