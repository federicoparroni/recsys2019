from extract_features.actions_involving_impression_session import ActionsInvolvingImpressionSession
from extract_features.average_impression_pos_interacted import ImpressionPositionInteracted
from extract_features.average_price_and_position_interaction import MeanPriceClickout
#from extract_features.frenzy_factor_consecutive_steps import FrenzyFactorSession
from extract_features.impression_features import ImpressionFeature
from extract_features.impression_position_session import ImpressionPositionSession
from extract_features.impression_price_info_session import ImpressionPriceInfoSession
from extract_features.item_popularity_session import ItemPopularitySession
from extract_features.label import ImpressionLabel
from extract_features.last_action_involving_impression import LastInteractionInvolvingImpression
from extract_features.mean_price_clickout import MeanPriceClickout_edo
#from extract_features.price_position_info_interactions import PricePositionInfoInteractedReferences
from extract_features.reference_position_in_next_clickout_impressions import ReferencePositionInNextClickoutImpressions
from extract_features.session_actions_num_ref_diff_from_impressions import SessionActionNumRefDiffFromImpressions
from extract_features.session_device import SessionDevice
from extract_features.session_filters_active_when_clickout import SessionFilterActiveWhenClickout
from extract_features.session_length import SessionLength
from extract_features.session_sort_order_when_clickout import SessionSortOrderWhenClickout
from extract_features.time_from_last_action_before_clk import TimePassedBeforeClickout
from extract_features.times_user_interacted_with_impression import TimesUserInteractedWithImpression
from extract_features.timing_from_last_interaction_impression import TimingFromLastInteractionImpression
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import data
import pandas as pd
from tqdm.auto import tqdm
import math
from utils.check_folder import check_folder
from multiprocessing import Process, Queue
tqdm.pandas()


def build_dataset(mode, features_array, cluster='no_cluster'):

    def func(x):
        y = x[x.action_type == 'clickout item']
        if len(y) > 0:
            return y.tail(1)

    def construct_features(df, q):
        dataset = df.groupby(['user_id', 'session_id']).progress_apply(
            func).reset_index(drop=True)
        dataset = dataset[['user_id', 'session_id', 'impressions']]
        dataset.impressions = dataset.impressions.str.split('|')
        dataset = dataset.impressions.apply(pd.Series) \
            .merge(dataset, right_index=True, left_index=True) \
            .drop(["impressions"], axis=1) \
            .melt(id_vars=['user_id', 'session_id'], value_name="item_id") \
            .drop("variable", axis=1) \
            .dropna()
        # dataset = dataset.sort_values(
        #     by=['user_id', 'session_id']).reset_index(drop=True)
        dataset.item_id = pd.to_numeric(dataset.item_id)

        # at this time we have session_id | user_id | item_id
        # now we should merge the features

        # load in the array all the features
        pandas_dataframe_features_list = []
        for f in features_array:
            pandas_dataframe_features_list.append(f(mode=mode, cluster=cluster).read_feature(one_hot=True))

        # merge the features on the dataframe
        for i in range(len(pandas_dataframe_features_list)):
            dataset = pd.merge(dataset, pandas_dataframe_features_list[i], how='inner')

        q.put(dataset)

    def save_features(dataset, count_chunk, target_session_id, target_user_id):
        test = dataset[dataset['user_id'].isin(
            target_user_id) & dataset['session_id'].isin(target_session_id)]
        train = dataset[(dataset['user_id'].isin(
            target_user_id) & dataset['session_id'].isin(target_session_id)) == False]

        if count_chunk == 1:
            path = 'dataset/preprocessed/{}/{}/tf-tanking/classification_train.csv'.format(
                cluster, mode)
            check_folder(path)
            train.to_csv(path)

            path = 'dataset/preprocessed/{}/{}/tf-ranking/classification_test.csv'.format(
                cluster, mode)
            check_folder(path)
            test.to_csv(path)
        else:
            with open('dataset/preprocessed/{}/{}/tf-ranking/classification_train.csv'.format(cluster, mode), 'a') as f:
                train.to_csv(f, header=False)
            with open('dataset/preprocessed/{}/{}/tf-ranking/classification_test.csv'.format(cluster, mode), 'a') as f:
                test.to_csv(f, header=False)

        print('chunk {} over {} completed'.format(count_chunk, math.ceil(
            len(session_indices)/session_to_consider_in_chunk)))

    train = data.train_df(mode=mode, cluster=cluster)
    test = data.test_df(mode=mode, cluster=cluster)
    target_indices = data.target_indices(mode=mode, cluster=cluster)
    target_user_id = test.loc[target_indices]['user_id'].values
    target_session_id = test.loc[target_indices]['session_id'].values

    full = pd.concat([train, test])
    del train
    del test

    # build in chunk
    # avoid session truncation, explicitly specify how many session you want in a chunk
    count_chunk = 0
    session_to_consider_in_chunk = 2000
    full = full.reset_index(drop=True)
    session_indices = list(
        full[['user_id']].drop_duplicates(keep='last').index.values)

    # create the dataset in parallel threads: one thread saves, the other create the dataset for the actual group
    p2 = None
    lower_index = full.head(1).index.values[0]
    session_indices_to_iterate_in = session_indices[session_to_consider_in_chunk:len(
        session_indices):session_to_consider_in_chunk]
    session_indices_to_iterate_in.append(
        session_indices[-1]) if session_indices[-1] != session_indices_to_iterate_in[-1] else session_indices_to_iterate_in
    for idx in session_indices_to_iterate_in:
        gr = full.loc[lower_index:idx]
        lower_index = idx + 1
        q = Queue()
        p1 = Process(target=construct_features, args=(gr, q,))
        p1.start()
        if p2 != None:
            p2.join()
        features = q.get()
        p1.join()
        count_chunk += 1
        if len(features > 0):
            p2 = Process(target=save_features, args=(
                features, count_chunk, target_session_id, target_user_id,))
            p2.start()


if __name__ == "__main__":
    features_array = [ImpressionLabel, ImpressionPriceInfoSession, LastInteractionInvolvingImpression,
                    TimingFromLastInteractionImpression, ImpressionPositionSession, ActionsInvolvingImpressionSession,
                    TimesUserInteractedWithImpression, ItemPopularitySession,MeanPriceClickout, MeanPriceClickout_edo,
                    SessionLength, SessionDevice, SessionActionNumRefDiffFromImpressions,
                    ImpressionPositionInteracted, ReferencePositionInNextClickoutImpressions,SessionFilterActiveWhenClickout,
                    SessionSortOrderWhenClickout,TimePassedBeforeClickout]
    build_dataset(mode='small', features_array=features_array)
