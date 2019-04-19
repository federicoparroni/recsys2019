import data
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()
import numpy as np
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


class NeuralNetworks(RecommenderBase):

    def __init__(self, mode, cluster, nn_dict_params):
        name = 'NeuralNetwork_2'
        super(NeuralNetworks, self).__init__(mode=mode, cluster=cluster, name=name)

        self.nn_dict_params = nn_dict_params
        base_path_dataset = f'dataset/preprocessed/neural_network_dataset/{cluster}/{mode}'

        self.X_train = np.load(f'{base_path_dataset}/X_train.npy')
        self.Y_train = np.load(f'{base_path_dataset}/Y_train.npy')

        self.X_val = np.load(f'{base_path_dataset}/X_val.npy')
        self.Y_val = np.load(f'{base_path_dataset}/Y_val.npy')

        self.class_weights_dict = None
        self._compute_class_weights()
        self._create_model()

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
        base_path = f'dataset/preprocessed/neural_network_dataset/{self.cluster}/{self.mode}'
        X = np.load(f'{base_path}/X_test.npy')
        target_indeces = np.load(f'{base_path}/target_indeces.npy')
        print(target_indeces)

        predictions = self.model.predict(X)

        final_predictions = []

        count = 0
        accumulator = 0
        for index in tqdm(target_indeces):
            impr = list(map(int, data.full_df().loc[index]['impressions'].split('|')))
            pred = predictions[accumulator:accumulator + len(impr)]
            accumulator += len(impr)
            couples = list(zip(pred, impr))
            couples.sort(key=lambda x: x[0], reverse=True)
            _, sorted_impr = zip(*couples)
            final_predictions.append((index, list(sorted_impr)))
            count += 1

        return final_predictions

    def get_scores_batch(self):
        pass

    def _create_model(self):
        model = Sequential()
        model.add(Dense(self.nn_dict_params['neurons_per_layer'], input_dim=self.X_train.shape[1],
                        activation=self.nn_dict_params['activation_function_internal_layers']))
        model.add(Dense(self.nn_dict_params['neurons_per_layer'],
                        activation=self.nn_dict_params['activation_function_internal_layers']))
        model.add(Dense(1, activation='sigmoid'))

        # compile the model
        model.compile(loss=self.nn_dict_params['loss'], optimizer=self.nn_dict_params['optimizer'],
                      metrics=['accuracy'])
        self.model = model
        print('model created')


def create_dataset_for_neural_networks(mode, cluster, columns_name_array):
    SAVE_PATH = f'dataset/preprocessed/neural_network_dataset/{cluster}/{mode}'
    #READ_PATH = f'dataset/preprocessed/FFNN_dataset/dataframes/{cluster}/{mode}/'
    check_folder.check_folder(SAVE_PATH)

    #train_df = pd.read_csv(f'{READ_PATH}/train_df.csv')
    #test_df = pd.read_csv(f'{READ_PATH}/test_df.csv')

    #train_df = pd.read_csv(f'{READ_PATH}/train_df.csv')
    #test_df = pd.read_csv(f'{READ_PATH}/test_df.csv')

    READ_PATH = f'dataset/preprocessed/{cluster}/{mode}/'

    train_df = pd.read_csv(f'{READ_PATH}/classification_train_xgboost.csv')
    test_df = pd.read_csv(f'{READ_PATH}/classification_test_xgboost.csv')


    """
    CREATE DATA FOR TRAIN

    """
    # the 5 column is the label
    X, Y = train_df.iloc[:, columns_name_array], train_df.iloc[:, [3]]
    scaler = MinMaxScaler()
    # normalize the values
    X_norm = scaler.fit_transform(X)
    Y_norm = Y.values

    X_train, X_val, Y_train, Y_val = train_test_split(X_norm, Y_norm, test_size=0.2, shuffle=True)


    print('saving training data...')
    np.save(f'{SAVE_PATH}/X_train', X_train)
    np.save(f'{SAVE_PATH}/X_val', X_val)
    print('done')

    print('saving labels...')
    np.save(f'{SAVE_PATH}/Y_train', Y_train)
    np.save(f'{SAVE_PATH}/Y_val', Y_val)
    print('done')

    """
    CREATE DATA FOR TEST

    """
    test_df.sort_values(['index', 'impression_position'], ascending=[True, True], inplace=True)
    X_test = test_df.iloc[:, columns_name_array]
    scaler = MinMaxScaler()

    # retrieving target indeces
    target_indices_duplicated = test_df['index']
    target_indeces = target_indices_duplicated.unique()
    # for i in range(0, target_indices_duplicated.shape[0], 25):
    #    target_indeces.append(target_indices_duplicated[i])

    target_indeces = np.array(target_indeces)

    # normalize the values
    X_test_norm = scaler.fit_transform(X_test)
    print('saving training data...')
    np.save(f'{SAVE_PATH}/X_test', X_test_norm)
    np.save(f'{SAVE_PATH}/target_indeces', target_indeces)
    print('done')


if __name__ == '__main__':
    columns = np.concatenate((np.array(range(4, 25)), np.array(range(26, 80))))
    #columns = np.array(range(4, 25))
    create_dataset_for_neural_networks('small', 'no_cluster', columns)

    nn_dict_params = {
        'activation_function_internal_layers': 'relu',
        'neurons_per_layer': 128,
        'loss': 'binary_crossentropy',
        'optimizer': 'adam',
        'validation_split': 0.2,
        'epochs': 10,
        'batch_size': 256,
    }

    model = NeuralNetworks(mode='small', cluster='no_cluster', nn_dict_params=nn_dict_params)
    model.evaluate()

    # out.create_sub(recs, 'prova')







