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

from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
from preprocess_utils.last_clickout_indices import expand_impressions
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
from extract_features.platform_reference_percentage_of_clickouts import PlatformReferencePercentageOfClickouts
from extract_features.platform_reference_percentage_of_interactions import PlatformReferencePercentageOfInteractions
from extract_features.location_reference_percentage_of_clickouts import LocationReferencePercentageOfClickouts
from extract_features.location_reference_percentage_of_interactions import LocationReferencePercentageOfInteractions
from extract_features.city_session import CitySession
from extract_features.perc_click_per_impressions import PercClickPerImpressions
from extract_features.platform_session import PlatformSession
from extract_features.past_future_session_features import PastFutureSessionFeatures




def dump_svmlight(df, save_path, save_num_features_path):
    print(len(df['index'].unique()))
    qid = df['index'].values
    print(f'shape data before dropping...{df.shape}')
    X, Y = df.drop(['session_id', 'user_id', 'label', 'item_id', 'index'], axis=1), df['label']
    del df
    # scaler = MinMaxScaler(feature_range=(-1, 1), copy=False)
    scaler = MaxAbsScaler(copy=False)
    # normalize the values
    X = scaler.fit_transform(X)
    Y_norm = Y.values
    del Y
    print(f'shape of the final data:{X.shape}')
    print(f'SAVING NUM FEATURES... \n {save_num_features_path}')
    with open(f'{save_num_features_path}/features_num.txt', 'w+') as f:
        f.write(f'{X.shape[1]}')
    print(f'SAVING DATA... \n {save_path}')
    dump_svmlight_file(X, Y_norm, save_path, query_id=qid, zero_based=False)
    print('DONE')

def merge_features_tf(mode, cluster, features_array):

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
    context_features_id = []
    for f in features_array:
        feature = f(mode=mode, cluster='no_cluster').read_feature(one_hot=True)
        # check if it is a feature of the impression
        if 'item_id' not in feature.columns:
            for i in range(train_df.shape[1]-6+1, train_df.shape[1]-6+1+feature.shape[1]-2, 1):
                context_features_id.append(str(i))
        print(f'session features names:{context_features_id}')
        print(f'shape of feature: {feature.shape}')
        print(f'len of feature:{len(feature)}')
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
    return train_df, validation_test_df, np.array(context_features_id)

def create_dataset(mode, cluster, features_array, dataset_name):
    _SAVE_BASE_PATH = f'dataset/preprocessed/tf_ranking/{cluster}/{mode}/{dataset_name}'
    cf.check_folder(_SAVE_BASE_PATH)
    train_df, vali_test_df, context_features_id =merge_features_tf(mode, cluster, features_array)

    # save context features id
    print(f'saving context feature id to: {_SAVE_BASE_PATH}/context_features_id.npy')
    np.save(f'{_SAVE_BASE_PATH}/context_features_id', context_features_id)

    dump_svmlight(train_df, f'{_SAVE_BASE_PATH}/train.txt', _SAVE_BASE_PATH)
    if mode == 'full':
        dump_svmlight(vali_test_df, f'{_SAVE_BASE_PATH}/test.txt', _SAVE_BASE_PATH)
    else:
        dump_svmlight(vali_test_df, f'{_SAVE_BASE_PATH}/vali.txt', _SAVE_BASE_PATH)
    print('PROCEDURE ENDED CORRECTLY')


if __name__ == '__main__':
    features_array = [ChangeImpressionOrderPositionInSession, TopPopPerImpression, TopPopInteractionClickoutPerImpression,
                      LocationReferencePercentageOfClickouts, LocationReferencePercentageOfInteractions, PlatformReferencePercentageOfInteractions,
                      PlatformReferencePercentageOfClickouts, PastFutureSessionFeatures, CitySession, SessionsImpressionsCountNumeric,
                      PlatformSession, FrenzyFactorSession, ImpressionRatingNumeric, ActionsInvolvingImpressionSession, ImpressionLabel,
                      ImpressionPriceInfoSession, SessionActionNumRefDiffFromImpressions,TimingFromLastInteractionImpression,
                      TimesUserInteractedWithImpression, ImpressionPositionSession, LastInteractionInvolvingImpression, SessionDevice,
                      SessionSortOrderWhenClickout, MeanPriceClickout, PricePositionInfoInteractedReferences, SessionLength, TimeFromLastActionBeforeClk,
                      TimesImpressionAppearedInClickoutsSession]

    """
    features_array = [ActionsInvolvingImpressionSession, ImpressionLabel, ImpressionPriceInfoSession,
                      TimingFromLastInteractionImpression, TimesUserInteractedWithImpression,
                      ImpressionPositionSession, LastInteractionInvolvingImpression,
                      TimesImpressionAppearedInClickoutsSession, MeanPriceClickout, SessionLength,
                      TimeFromLastActionBeforeClk, PricePositionInfoInteractedReferences,
                      SessionDevice, ActionTypeBefClick, ImpressionRating, SessionsImpressionsCountNumeric,
                      ChangeImpressionOrderPositionInSession,User2Item,User2Item,User2Item]
    """


#StepsBeforeLastClickout
#ChangeOfSortOrderBeforeCurrent
    print('insert mode:')
    mode = input()
    print('insert cluster name:')
    cluster = input()
    print('insert dataset_name:')
    dataset_name = input()

    create_dataset(mode, cluster, features_array, dataset_name)
