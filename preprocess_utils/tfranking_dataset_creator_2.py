import data
from functools import reduce
import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from preprocess_utils.last_clickout_indices import find as find_last_clickout_indices
from preprocess_utils.last_clickout_indices import expand_impressions
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import utils.check_folder as cf
import utils.menu as menu
from sklearn.model_selection import KFold
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


def dump_hdf(df, save_path, save_num_features_path):
    print(len(df['index'].unique()))
    print(f'shape data before dropping...{df.shape}')
    df.rename(columns={'index': 'qid'}, inplace=True)
    X = df.drop(['session_id', 'user_id', 'item_id'], axis=1)

    #save the columns names
    columns = X.columns

    del df
    # scaler = MinMaxScaler(feature_range=(-1, 1), copy=False)
    scaler = MaxAbsScaler(copy=False)
    # normalize the values
    X = scaler.fit_transform(X)

    X = pd.DataFrame(X, columns=columns)
    print(f'shape of the final data:{X.shape}')
    print(f'SAVING NUM FEATURES... \n {save_num_features_path}')
    with open(f'{save_num_features_path}/features_num.txt', 'w+') as f:
        f.write(f'{X.shape[1]}')
    print(f'SAVING DATA... \n {save_path}')
    X.to_hdf(save_path, key='df', index=False)
    print('DONE')

def merge_features_tf_cv(mode, cluster, features_array):
    # load the full_df
    train_df = data.train_df(mode, cluster)
    test_df = data.test_df(mode, cluster)
    full_df = pd.concat([train_df, test_df])
    del train_df, test_df

    # retrieve the indeces of the last clikcouts
    print('find_last_click_idxs')
    last_click_idxs = find_last_clickout_indices(full_df)

    # filter on the found indeces obtaining only the rows of a last clickout
    print('filter full on last click idxs')
    click_df = full_df.loc[last_click_idxs].copy()

    # expand the impression as rows
    print('expand the impression')
    click_df = expand_impressions(click_df)[['user_id', 'session_id', 'item_id', 'index']]
    click_df['dummy_step'] = np.arange(len(click_df))

    # do the join
    print('join with the features')
    print(f'train_shape: {click_df.shape}\n')
    context_features_id = []
    for f in features_array:
        if type(f) == tuple:
            feature = f[0](mode=mode, cluster='no_cluster').read_feature(one_hot=f[1])
        else:
            feature = f(mode=mode, cluster='no_cluster').read_feature(one_hot=True)
        print(f'columns of the feature:\n {feature.columns}')
        print(f'NaN values are: {feature.isnull().values.sum()}')
        # if there are none fill it with -1
        feature.fillna(0, inplace=True)
        # check if it is a feature of the impression
        if 'item_id' not in feature.columns:
            for i in range(click_df.shape[1] - 6 + 1, click_df.shape[1] - 6 + 1 + feature.shape[1] - 2, 1):
                context_features_id.append(str(i))
        print(f'session features names:{context_features_id}')
        print(f'shape of feature: {feature.shape}')
        print(f'len of feature:{len(feature)}')
        click_df = click_df.merge(feature)
        print(f'train_shape: {click_df.shape}\n ')

    print('sorting by index and step...')
    # sort the dataframes
    click_df.sort_values(['index', 'dummy_step'], inplace=True)
    click_df.drop('dummy_step', axis=1, inplace=True)

    print('after join')
    return click_df, np.array(context_features_id)
    
    
    

def merge_features_tf(mode, cluster, features_array, stacking_scores_path):

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
        if type(f) == tuple:
            feature = f[0](mode=mode, cluster='no_cluster').read_feature(one_hot=f[1])
        else:
            feature = f(mode=mode, cluster='no_cluster').read_feature(one_hot=True)
        print(f'columns of the feature:\n {feature.columns}')
        print(f'NaN values are: {feature.isnull().values.sum()}')
        # if there are none fill it with -1
        feature.fillna(-1, inplace=True)
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

    if len(stacking_scores_path)>1:
        for path in stacking_scores_path:
            score = pd.read_csv(path)
            cols = [c for c in score.columns if c in ['user_id', 'session_id', 'item_id'] or 'score' in c]
            score = score[cols]
            #if 'rnn' in path:
            score = score.groupby(['user_id', 'session_id', 'item_id'], as_index=False).last()
            train_df = train_df.merge(score, on=['user_id', 'session_id', 'item_id'], how='left')
            validation_test_df = validation_test_df.merge(score, on=['user_id', 'session_id', 'item_id'], how='left')
            print(f'train_shape: {train_df.shape}\n vali_shape: {validation_test_df.shape}')

    train_df.fillna(0, inplace=True)
    validation_test_df.fillna(0, inplace=True)

    print('sorting by index and step...')
    # sort the dataframes
    train_df.sort_values(['index', 'dummy_step'], inplace=True)
    train_df.drop('dummy_step', axis=1, inplace=True
                  )
    validation_test_df.sort_values(['index', 'dummy_step'], inplace=True)
    validation_test_df.drop('dummy_step', axis=1, inplace=True)

    print('after join')
    return train_df, validation_test_df, np.array(context_features_id)

def create_dataset_cv(mode, cluster, features_array, dataset_name, k):

    # create the folders for the cv split
    _SAVE_BASE_PATH = f'dataset/preprocessed/tf_ranking/{cluster}/{mode}/{dataset_name}'
    cf.check_folder(_SAVE_BASE_PATH)

    # retrieve the dataframe
    train_df, context_features_id = merge_features_tf_cv(mode, cluster, features_array)
    train_df.rename(columns={'index': 'qid'}, inplace=True)
    columns = train_df.columns

    # save context features id
    print(f'saving context feature id to: {_SAVE_BASE_PATH}/context_features_id.npy')
    np.save(f'{_SAVE_BASE_PATH}/context_features_id', context_features_id)

    # computing groups len
    print('retrieving groups len')
    groups_len = train_df[['qid', 'label']].groupby('qid').count().values.flatten()

    # reshape in a way that on a row we have a session
    reshaped_array = np.split(train_df.values, np.cumsum(groups_len)[:-1])
    del train_df

    # compute the folds
    kf = KFold(k)
    fold = 1
    for train_idxs, test_idxs in (kf.split(reshaped_array)):

        train_array = np.array(reshaped_array)[train_idxs]
        test_array = np.array(reshaped_array)[test_idxs]

        train_df = pd.DataFrame(np.concatenate(train_array), columns=columns)
        test_df = pd.DataFrame(np.concatenate(test_array), columns=columns)

        save_path = f'{_SAVE_BASE_PATH}/fold_{fold}'
        cf.check_folder(save_path)

        parse_dataset(train_df, save_path, 'train')
        parse_dataset(test_df, save_path, 'test')

        fold += 1


def parse_dataset(df, save_path, mode):
    assert mode in ['train', 'test']

    def split_and_pad(a, group_lengths, pad_len=25, pad_value=0, _dtype=np.float32):
        col_len = a.shape[1]
        # build a list of np array with splitted groups
        splitted = np.split(a, np.cumsum(group_lengths)[:-1])
        # pad = pad_len - group_lengths
        res = np.zeros((len(group_lengths), pad_len, col_len), dtype=_dtype)
        print(res.shape)
        for i, r in enumerate(splitted):
            g = (np.ones((pad_len, col_len)) * pad_value).astype(_dtype)
            g[0:group_lengths[i], :] = splitted[i]
            res[i] = g
        return res

    df.rename(columns={'index': 'qid'}, inplace=True)

    # computing groups len
    print('retrieving groups len')
    groups_len = df[['qid', 'label']].groupby('qid').count().values.flatten()

    if mode == 'test':
        # save usi df
        usi_df = df[['session_id', 'user_id', 'item_id']]
        usi_array = split_and_pad(usi_df.values, groups_len, pad_value=0, _dtype=object)
        usi_df = pd.DataFrame(usi_array.reshape((-1, usi_df.shape[1])), columns=usi_df.columns)
        usi_df.to_csv(f'{save_path}/usi.csv')

    # drop user session item
    df = df.drop(['session_id', 'user_id', 'item_id'], axis=1)

    # retrieve the label
    label = df.pop('label')
    qid = df.pop('qid')

    print(f'SAVING NUM FEATURES... \n {save_path}')
    with open(f'{save_path}/features_num.txt', 'w+') as f:
        f.write(f'{df.shape[1]}')

    # substitute the -1 with 0
    df.replace(-1, 0, inplace=True)

    # normalize the data
    scaler = MaxAbsScaler(copy=False)
    df = scaler.fit_transform(df)

    # pad the sessions
    print('padding...')
    tensor_padded = split_and_pad(df, groups_len, pad_value=0)
    label_padded = split_and_pad(label.values.reshape(-1, 1), groups_len, pad_value=0)
    print('done')

    # reshape
    print('reshaping...')
    label_list = label_padded.reshape(-1, 25)
    feature_map = {str(i): tensor_padded[:, :, i].reshape(-1, 25, 1) for i in tqdm(range(tensor_padded.shape[-1]))}

    print('saving label list')
    np.save(f'{save_path}/label_list_{mode}', label_list)
    print('saving feature map')
    np.save(f'{save_path}/feature_map_{mode}', feature_map)
    print('procedure done!')


def create_dataset(mode, cluster, features_array, dataset_name, stacking_scores_path):
    _SAVE_BASE_PATH = f'dataset/preprocessed/tf_ranking/{cluster}/{mode}/{dataset_name}'
    cf.check_folder(_SAVE_BASE_PATH)
    train_df, vali_test_df, context_features_id = merge_features_tf(mode, cluster, features_array, stacking_scores_path)

    # save context features id
    print(f'saving context feature id to: {_SAVE_BASE_PATH}/context_features_id.npy')
    np.save(f'{_SAVE_BASE_PATH}/context_features_id', context_features_id)

    parse_dataset(train_df, _SAVE_BASE_PATH, 'train')
    parse_dataset(vali_test_df, _SAVE_BASE_PATH, 'test')
    Hera.send_message('tf ranking dataset saved !')
    print('PROCEDURE ENDED CORRECTLY')


if __name__ == '__main__':


    features_array = [
        ImpressionLabel,
        (StatisticsPosInteracted, False),
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
        (ImpressionPriceInfoSession, False),
        (ImpressionRatingNumeric, False),
        (ImpressionStarsNumeric, False),
        ##LastInteractionInvolvingImpression,
        LastClickoutFiltersSatisfaction,
        (StepsBeforeLastClickout,False),
        (LazyUser,False),
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
        #SessionFilterActiveWhenClickout,
        SessionLength,
        # SessionNumClickouts,
        # SessionNumFilterSel,
        # SessionNumInterItemImage,
        # SessionNumNotNumeric,
        SessionSortOrderWhenClickout,
        (StatisticsTimeFromLastAction, False),
        TimePerImpression,
        TimesUserInteractedWithImpression,
        TimingFromLastInteractionImpression,
        TopPopInteractionClickoutPerImpression,
        TopPopPerImpression,
        User2Item,
        #UserFeature
    ]
    



    assert features_array[0] == ImpressionLabel, 'first feature must be the label!'


    choice = menu.yesno_choice(['want the scores?'])
    if choice == 'y':
        base_path_stacking = 'scores_stacking'
        stacking_scores_path = ['xgboost_nobias.csv.gz', 'catboost_rank.csv.gz','rnn_GRU_2layers_64units_2dense_class_nobias_05952.csv.gz',
                                'scores_pairwise_soft_zero_one_loss.csv.gz']
        stacking_scores_path = [f'{base_path_stacking}/{a}' for a in stacking_scores_path]
    else:
        stacking_scores_path = []


    mode = menu.mode_selection()
    cluster = menu.cluster_selection()
    dataset_name = input('insert dataset name\n')

    choice = menu.single_choice(['select mode'], ['normal', 'cv'])
    if choice == 'cv':
        create_dataset_cv(mode, cluster, features_array, dataset_name, k=5)
    else:
        create_dataset(mode, cluster, features_array, dataset_name, stacking_scores_path)
