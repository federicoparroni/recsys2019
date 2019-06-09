import data
from extract_features.classifier.avg_interacted_price import AvgInteractedPrice
from extract_features.classifier.first_impression_price_info import FirstImpressionPriceInfo
from extract_features.classifier.location_reference_first_impression import LocationReferenceFirstImpression
from extract_features.classifier.num_interactions_with_first_impression_in_history import \
    NumInteractionsWithFirstImpressionInHistory
from extract_features.classifier.actions_count_classifier import ActionsCountClassifier
from extract_features.classifier.platform_reference_first_impression import PlatformReferenceFirstImpression
from extract_features.classifier.platfrom import Platform
from extract_features.classifier.popularity_first_impression import PopularityFirstImpression
from extract_features.classifier.first_impression_price import FirstImpressionPrice
from extract_features.classifier.price_stats import PriceStats
from extract_features.classifier.rnn_output import RNNOutput
from extract_features.classifier.label_classification import LabelClassification
from extract_features.classifier.last_action_before_clickout import LastActionBeforeClickout
from extract_features.classifier.last_action_involving_first_impression import LastActionInvolvingFirstImpressions
from extract_features.classifier.user_feature_first_impression import UserFeatureFirstImpression
from extract_features.num_impressions_in_clickout import NumImpressionsInClickout
from extract_features.classifier.num_interactions_with_first_impression import NumInteractionsWithFirstImpression
from extract_features.price_position_info_interactions import PricePositionInfoInteractedReferences
from extract_features.session_device import SessionDevice
from extract_features.session_length import SessionLength
from extract_features.session_sort_order_when_clickout import SessionSortOrderWhenClickout
from extract_features.time_from_last_action_before_clk import TimeFromLastActionBeforeClk
from extract_features.classifier.popularity_clickout_first_impression import PopularityClickoutFirstImpression
from extract_features.classifier.stars_ratings_first_impression import StarsRatingsFirstImpression
from extract_features.frenzy_factor_consecutive_steps import FrenzyFactorSession
from extract_features.session_actions_num_ref_diff_from_impressions import SessionActionNumRefDiffFromImpressions
from extract_features.classifier.timing_from_last_interaction_first_impression import TimingFromLastInteractionFirstImpression
from extract_features.day_moment_in_day import DayOfWeekAndMomentInDay
from utils.check_folder import check_folder


def merge_features_classifier(mode, cluster, features_array, starting_feature):
    df = starting_feature(mode=mode, cluster=cluster).read_feature()
    for f in features_array:
        feature = f(mode=mode, cluster=cluster)
        df = feature.join_to(df, one_hot=True)
        print("Merged with feature:" + feature.name)
        print("New df shape: {}".format(df.shape))

    test_df = data.test_df(mode, cluster)
    test_df = test_df[(test_df.action_type == "clickout item") & (test_df.reference.isnull())]
    sessions = set(test_df.session_id)
    train_df = df[~df.session_id.isin(sessions)]
    test_df = df[df.session_id.isin(sessions)]
    return train_df, test_df

def create_dataset(mode, cluster):
    # training
    features_array = [SessionDevice,
                      SessionSortOrderWhenClickout,
                      PricePositionInfoInteractedReferences,
                      SessionLength,
                      TimeFromLastActionBeforeClk,
                      LastActionBeforeClickout,
                      NumInteractionsWithFirstImpression,
                      FirstImpressionPrice,
                      LastActionInvolvingFirstImpressions,
                      NumImpressionsInClickout,
                      #Platform,
                      RNNOutput,
                      PriceStats,
                      PopularityFirstImpression,
                      AvgInteractedPrice,
                      PopularityClickoutFirstImpression,
                      LocationReferenceFirstImpression,
                      PlatformReferenceFirstImpression,
                      FrenzyFactorSession,
                      StarsRatingsFirstImpression,
                      ActionsCountClassifier,
                      FirstImpressionPriceInfo,
                      SessionActionNumRefDiffFromImpressions,
                      TimingFromLastInteractionFirstImpression,
                      DayOfWeekAndMomentInDay,
                      UserFeatureFirstImpression
                      ]

    train_df, test_df = merge_features_classifier(mode, cluster, features_array, LabelClassification)
    check_folder('dataset/preprocessed/{}/{}/xgboost_classifier/'.format(cluster, mode))

    train_df.to_csv('dataset/preprocessed/{}/{}/xgboost_classifier/train.csv'.format(cluster, mode), index=False)
    test_df.to_csv('dataset/preprocessed/{}/{}/xgboost_classifier/test.csv'.format(cluster, mode), index=False)

    print("Dataset created!")


if __name__ == "__main__":
    from utils.menu import mode_selection
    mode = mode_selection()
    cluster = 'no_cluster'
    create_dataset(mode, cluster)
