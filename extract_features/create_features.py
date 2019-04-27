import utils.menu as menu
from extract_features.actions_involving_impression_session import ActionsInvolvingImpressionSession
from extract_features.mean_price_clickout import MeanPriceClickout
from extract_features.label import ImpressionLabel
from extract_features.impression_position_session import ImpressionPositionSession
from extract_features.session_length import SessionLength
from extract_features.session_device import SessionDevice
from extract_features.session_filters_active_when_clickout import SessionFilterActiveWhenClickout
from extract_features.session_sort_order_when_clickout import SessionSortOrderWhenClickout
from extract_features.impression_price_info_session import ImpressionPriceInfoSession
from extract_features.times_user_interacted_with_impression import TimesUserInteractedWithImpression
from extract_features.timing_from_last_interaction_impression import TimingFromLastInteractionImpression
from extract_features.last_action_involving_impression import LastInteractionInvolvingImpression
from extract_features.session_actions_num_ref_diff_from_impressions import SessionActionNumRefDiffFromImpressions

if __name__ == "__main__":    
    mode = menu.mode_selection()
    cluster = menu.cluster_selection()

    # define all the features to be made
    features = [ActionsInvolvingImpressionSession, MeanPriceClickout, ImpressionLabel, ImpressionPositionSession,
                SessionLength, SessionDevice, SessionFilterActiveWhenClickout, SessionSortOrderWhenClickout,
                ImpressionPriceInfoSession, TimesUserInteractedWithImpression, TimingFromLastInteractionImpression,
                LastInteractionInvolvingImpression, SessionActionNumRefDiffFromImpressions]

    # create all the features defined in the array 'features'
    for feature_name in features:
        feature = feature_name(mode=mode, cluster=cluster)
        feature.save_feature()
