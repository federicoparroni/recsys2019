import data
import pandas as pd
from tqdm.auto import tqdm
import pickle

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
from extract_features.impression_features import ImpressionFeature
from extract_features.item_popularity_session import ItemPopularitySession


tqdm.pandas()
import numpy as np
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight

pd.options.display.max_columns = None
from keras.callbacks import EarlyStopping
from recommenders.recommender_base import RecommenderBase

import utils.check_folder as check_folder
from keras import metrics
import keras.optimizers
from keras import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout
import out
from sklearn.model_selection import train_test_split
from extract_features.feature_base import FeatureBase


class NeuralNetworks(RecommenderBase):

    def __init__(self, mode, cluster, nn_dict_params):
        name = 'NeuralNetwork_2'
        super(NeuralNetworks, self).__init__(mode=mode, cluster=cluster, name=name)

        self.scores_batch = None

        self.nn_dict_params = nn_dict_params
        self.dataset_name = nn_dict_params['dataset_name']
        base_path_dataset = f'dataset/preprocessed/neural_network_dataset/{cluster}/{mode}/{self.dataset_name}'

        self.X_train = np.load(f'{base_path_dataset}/X_train.npy')
        self.Y_train = np.load(f'{base_path_dataset}/Y_train.npy')

        self.X_val = np.load(f'{base_path_dataset}/X_val.npy')
        self.Y_val = np.load(f'{base_path_dataset}/Y_val.npy')

        self.class_weights_dict = None
        self._compute_class_weights()
        #self._create_model()
        self._create_model_from_arr()

    def _compute_class_weights(self):
        temp = np.concatenate((self.Y_train, self.Y_val))
        #class_weights = class_weight.compute_class_weight('balanced', np.unique(temp), temp)


        class_weights_dict = {
            0: 1,
            1: (len(temp)-np.sum(temp))/np.sum(temp),
        }

        self.class_weights_dict=class_weights_dict


    def fit(self):
        callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None,
                                 restore_best_weights=True)
        self.model.fit(self.X_train, self.Y_train, validation_data=(self.X_val, self.Y_val),
                       epochs=self.nn_dict_params['epochs'],
                       batch_size=self.nn_dict_params['batch_size'],
                       shuffle=True,
                       class_weight=self.class_weights_dict)

    def recommend_batch(self):
        base_path = f'dataset/preprocessed/neural_network_dataset/{self.cluster}/{self.mode}/{self.dataset_name}'
        X = np.load(f'{base_path}/X_test.npy')
        target_indeces = np.load(f'{base_path}/target_indeces.npy')
        print(target_indeces)

        predictions = self.model.predict(X)
        predictions = [a[0] for a in predictions]

        final_predictions = []
        scores_batch = []

        count = 0
        accumulator = 0
        for index in tqdm(target_indeces):
            impr = list(map(int, data.full_df().loc[index]['impressions'].split('|')))
            pred = predictions[accumulator:accumulator + len(impr)]
            # threshold the prediction score
            #pred = [a if a > 0.8 else 0 for a in pred]

            accumulator += len(impr)
            couples = list(zip(pred, impr))

            print(couples[0])

            couples.sort(key=lambda x: x[0], reverse=True)

            print(couples[0])
            scores, sorted_impr = zip(*couples)
            final_predictions.append((index, list(sorted_impr)))
            scores_batch.append((index, list(sorted_impr), list(scores)))
            count += 1
        self.scores_batch = scores_batch
        return final_predictions

    def get_scores_batch(self, save=False):
        _ = self.recommend_batch()
        base_path = f'dataset/preprocessed/neural_network_dataset/{self.cluster}/{self.mode}/predictions/{self.dataset_name}.pickle'
        check_folder.check_folder(base_path)
        if save:
            with open(base_path, 'wb') as f:
                pickle.dump(self.scores_batch, f)
            print(f'saved at: {base_path}')
        else:
            return self.scores_batch

    def _create_model(self):
        model = Sequential()
        model.add(Dense(self.nn_dict_params['neurons_per_layer'], input_dim=self.X_train.shape[1],
                        activation=self.nn_dict_params['activation_function_internal_layers']))
        model.add(Dense(self.nn_dict_params['neurons_per_layer'],
                        activation=self.nn_dict_params['activation_function_internal_layers']))
        model.add(Dense(self.nn_dict_params['neurons_per_layer'],
                        activation=self.nn_dict_params['activation_function_internal_layers']))
        model.add(Dense(self.nn_dict_params['neurons_per_layer'],
                        activation=self.nn_dict_params['activation_function_internal_layers']))
        model.add(Dense(self.nn_dict_params['neurons_per_layer'],
                        activation=self.nn_dict_params['activation_function_internal_layers']))
        model.add(Dense(1, activation='sigmoid'))

        # compile the model
        model.compile(loss=self.nn_dict_params['loss'], optimizer=self.nn_dict_params['optimizer'],
                      metrics=['accuracy'])
        self.model = model
        print('model created')

    def _create_model_from_arr(self):
        """
        the array have to be composed by tuples, the possible tuples are

        NOTE the first element of the array have to be the nuber of neurons of the first dense layer!

        -Dense layer (#continguos equals layer, d, #neurons)
        -Dropout layer (#continguos equals layer, drop, rate)

        example:
        (3, d, 128) will create 3 layer dense with 128 neurons each

        [128, (2,d,128), (1, d, 64)] will create a network with 3 layers dense with 128 neurons 1 dense with
        64 neurons and a last layer dense with 1 neurons

        :param arr: array with network structure
        :return: -
        """
        arr = self.nn_dict_params['model_array']

        model = Sequential()
        for i in range(len(arr)):
            # the network is initialized with a dense layer
            if i == 0:
                model.add(Dense(arr[i], input_dim=self.X_train.shape[1],
                                activation=self.nn_dict_params['activation_function_internal_layers']))
            else:
                layers_number = arr[i][0]
                layer_type = arr[i][1]

                if layer_type == 'd':
                    for j in range(layers_number):
                        model.add(Dense(arr[i][2], activation=self.nn_dict_params['activation_function_internal_layers']))

                elif layer_type == 'drop':
                    for j in range(layers_number):
                        model.add(Dropout(rate=arr[i][2]))
                else:
                    print('not yet implemented!!!')
                    exit(0)
        # add at the end the last dense layer composed by only one neuron
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss=self.nn_dict_params['loss'], optimizer=self.nn_dict_params['optimizer'],
                      metrics=['accuracy'])
        self.model = model
        print('model created')


def is_target(df, tgt_usersession):
    if tuple(df.head(1)[['user_id', 'session_id']].values[0]) in tgt_usersession:
        return True
    else:
        return False


def create_dataset_for_neural_networks(mode, cluster, features_array, dataset_name):
    # path in which the computed dataset will be stored
    SAVE_PATH = f'dataset/preprocessed/neural_network_dataset/{cluster}/{mode}/{dataset_name}'
    check_folder.check_folder(SAVE_PATH)

    READ_FEATURE_PATH = f'dataset/preprocessed/{cluster}/{mode}/feature'

    """
    RETRIEVE THE FEATURES
    """
    ################################################

    # list of pandas dataframe each element represent a feature
    pandas_dataframe_features_session_list = []
    for f in features_array['session']:
        pandas_dataframe_features_session_list.append(f(mode=mode, cluster=cluster).read_feature(one_hot=True))

    # merge all the dataframes
    df_merged_session = reduce(lambda left, right: pd.merge(left, right, on=['user_id', 'session_id'],
                                                    how='inner'), pandas_dataframe_features_session_list)

    pandas_dataframe_features_item_list = []
    for f in features_array['item_id']:
        pandas_dataframe_features_item_list.append(f(mode=mode, cluster=cluster).read_feature(one_hot=True))

    # merge all the dataframes
    df_merged_item = reduce(lambda left, right: pd.merge(left, right, on=['user_id', 'session_id', 'item_id'],
                                                            how='inner'), pandas_dataframe_features_item_list)

    df_merged = pd.merge(df_merged_item, df_merged_session, on=['user_id', 'session_id'])

    ################################################


    # load the target indeces of the mode
    target_indeces = data.target_indices(mode, cluster)
    print(f'number of tgt index: {len(target_indeces)}')

    # load the full df
    full_df = data.full_df()

    # dict that has as keys the couples (user_id, session_id) that are target
    tgt_usersession = {}
    for index in target_indeces:
        tgt_usersession[tuple(full_df.iloc[index][['user_id', 'session_id']].values)] = index

    is_target_ = df_merged.groupby(['user_id', 'session_id']).progress_apply(is_target, tgt_usersession=tgt_usersession)
    df_merged = pd.merge(df_merged, is_target_.reset_index(), on=['user_id', 'session_id'])

    test_df = df_merged[df_merged[0]==True]
    train_df = df_merged[df_merged[0]==False]

    train_df.drop(columns=[0], inplace=True)
    test_df.drop(columns=[0], inplace=True)

    # retrieve the target indeces in the right order
    couples_dict = {}
    couples_arr = test_df[['user_id', 'session_id']].values
    for c in couples_arr:
        if tuple(c) not in couples_dict:
            couples_dict[tuple(c)] = 1

    target_us_reordered = list(couples_dict.keys())

    target_indeces_reordered = []
    for k in target_us_reordered:
        target_indeces_reordered.append(tgt_usersession[k])

    print(f'number of tgt index: {len(target_indeces_reordered)}')
    target_indeces_reordered = np.array(target_indeces_reordered)

    """
    CREATE DATA FOR TRAIN

    """
    # the 5 column is the label
    X, Y = train_df.iloc[:, 4:], train_df['label']
    scaler = MinMaxScaler()
    # normalize the values
    X_norm = scaler.fit_transform(X)
    Y_norm = Y.values

    X_train, X_val, Y_train, Y_val = train_test_split(X_norm, Y_norm, test_size=0.2, shuffle=False)

    X_test = test_df.iloc[:, 4:]
    X_test_norm = scaler.fit_transform(X_test)

    print('saving training data...')
    np.save(f'{SAVE_PATH}/X_train', X_train)
    np.save(f'{SAVE_PATH}/X_val', X_val)
    print('done')

    print('saving labels...')
    np.save(f'{SAVE_PATH}/Y_train', Y_train)
    np.save(f'{SAVE_PATH}/Y_val', Y_val)
    print('done')

    print('saving test data...')
    np.save(f'{SAVE_PATH}/X_test', X_test_norm)
    np.save(f'{SAVE_PATH}/target_indeces', target_indeces_reordered)
    print('done')


if __name__ == '__main__':
    features = {
        'item_id': [ImpressionLabel, ImpressionPriceInfoSession, LastInteractionInvolvingImpression, TimingFromLastInteractionImpression],
        'session': [MeanPriceClickout, SessionLength, SessionDevice]
    }

    dataset_name = 'all'
    #create_dataset_for_neural_networks('small', 'no_cluster', features, dataset_name)

    nn_dict_params = {
        'model_array': [256, (2, 'd', 128), (1, 'drop', 0.2), (2, 'd', 64), (1, 'drop', 0.2), (2, 'd', 32)],
        'dataset_name': 'all',
        'activation_function_internal_layers': 'relu',
        'neurons_per_layer': 256,
        #'loss': 'binary_crossentropy',
        'loss': 'mean_squared_error',
        'optimizer': 'adam',
        'validation_split': 0.2,
        'epochs': 1,
        'batch_size': 256,
    }

    model = NeuralNetworks(mode='small', cluster='no_cluster', nn_dict_params=nn_dict_params)
    #model.fit()
    #model.get_scores_batch(save=True)
    model.evaluate()
    # out.create_sub(recs, 'prova')







