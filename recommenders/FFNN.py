import data
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
pd.options.display.max_columns = None
from keras.callbacks import EarlyStopping
from recommenders.recommender_base import RecommenderBase

import utils.check_folder as check_folder
from keras import metrics
from keras import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout
import out
from sklearn.model_selection import train_test_split


class NeuralNetworks(RecommenderBase):

    def __init__(self, mode, cluster, nn_dict_params):
        name = 'NeuralNetwork'
        super(NeuralNetworks, self).__init__(mode=mode, cluster=cluster, name=name)

        self.nn_dict_params = nn_dict_params
        base_path_dataset = f'dataset/preprocessed/FFNN_dataset/{cluster}/{mode}'

        self.X_train = np.load(f'{base_path_dataset}/X_train.npy')
        self.Y_train = np.load(f'{base_path_dataset}/Y_train.npy')

        self.X_val = np.load(f'{base_path_dataset}/X_val.npy')
        self.Y_val = np.load(f'{base_path_dataset}/Y_val.npy')

        self._create_model()

    def _mrr_metric(self, y_true, y_pred):
        mrr = 0
        current_percentage = 0
        for i in range(1, 26, 1):
            if i == 1:
                mrr = metrics.top_k_categorical_accuracy(y_true, y_pred, k=i)
                current_percentage = metrics.top_k_categorical_accuracy(y_true, y_pred, k=i)
            else:
                t = metrics.top_k_categorical_accuracy(y_true, y_pred, k=i)
                mrr += (t - current_percentage) * (1 / i)
                current_percentage = t
        return mrr

    def fit(self):
        callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None,
                                 restore_best_weights=True)
        self.model.fit(self.X_train, self.Y_train, validation_data=(self.X_val, self.Y_val),
                       epochs=self.nn_dict_params['epochs'],
                       batch_size=self.nn_dict_params['batch_size'],
                       shuffle=True)

    def recommend_batch(self):
        base_path = f'dataset/preprocessed/FFNN_dataset/{self.cluster}/{self.mode}'
        X = np.load(f'{base_path}/X_test.npy')
        target_indeces = np.load(f'{base_path}/target_indeces.npy')

        predictions = self.model.predict(X)

        final_predictions = []

        count = 0
        for index in tqdm(target_indeces):
            impr = list(map(int, data.full_df().loc[index]['impressions'].split('|')))
            pred = predictions[count][0:len(impr)]
            couples = list(zip(pred, impr))
            print(couples)
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
        model.add(Dropout(rate=0.2))
        model.add(Dense(self.nn_dict_params['neurons_per_layer'],
                        activation=self.nn_dict_params['activation_function_internal_layers']))
        model.add(Dropout(rate=0.2))
        model.add(Dense(self.nn_dict_params['neurons_per_layer'],
                        activation=self.nn_dict_params['activation_function_internal_layers']))
        model.add(Dropout(rate=0.2))
        model.add(Dense(50,
                        activation=self.nn_dict_params['activation_function_internal_layers']))
        model.add(Dense(25, activation='softmax'))

        # compile the model
        model.compile(loss=self.nn_dict_params['loss'], optimizer=self.nn_dict_params['optimizer'], metrics=[metrics.categorical_accuracy, self._mrr_metric])
        self.model = model
        print('model created')


def _extract_features(df, submission_mode=False):
    # get the rows where the action is 'clickout item'
    if submission_mode:
        clickout_rows_df = df[(df['action_type'] == 'clickout item') & df['reference'].isnull()]
    else:
        clickout_rows_df = df[df['action_type'] == 'clickout item']

    if len(clickout_rows_df) > 0:

        # features
        features = {
            # impressions features
            'times_impression_appeared': [],
            'time_elapsed_from_last_time_impression_appeared': [],
            'steps_from_last_time_impression_appeared': [],
            'kind_action_reference_appeared': [],
            'impression_position': [],
            'label': [],
            'price': [],
            'price_position': [],

            'delta_position': [],

            # session features
            'session_length': [],
            'session_steps': [],
            'time_from_last_action': [],
            'reference_position_last_action': [],

            'index': []}

        clk = clickout_rows_df.tail(1)

        head_index = df.head(1).index

        # considering only the past!
        # mantain of the df only the actions before the last clickout
        df = df.loc[head_index.values[0]:clk.index.values[0] - 1]

        if len(df) > 0:
            session_length = clk['timestamp'].values[0] - df.head(1)['timestamp'].values[0]
            time_from_last_action = clk['timestamp'].values[0] - df.tail(1)['timestamp'].values[0]
            if df.tail(1)['reference'].values[0].isdigit():
                last_ref = int(df.tail(1)['reference'])
            else:
                last_ref = 0
        else:
            session_length = -0.5
            time_from_last_action = -0.5
            last_ref = -0.5
        session_steps = clk['step'].values[0]

        # get the impression
        impr = list(map(int, clk['impressions'].values[0].split('|')))

        if last_ref in impr:
            reference_position_last_action = impr.index(last_ref)
        else:
            reference_position_last_action = -0.5

        prices = list(map(int, clk['prices'].values[0].split('|')))
        sorted_prices = prices.copy()
        sorted_prices.sort()

        references = df['reference'].values

        count = 0
        for i in impr:
            if reference_position_last_action >= 0:
                delta_pos = count - reference_position_last_action
            else:
                delta_pos = count
            indices = np.where(references == str(i))[0]

            features['index'].append(clk.index[0])
            features['impression_position'].append(count + 1)
            features['price'].append(prices[count])
            features['price_position'].append(sorted_prices.index(prices[count]))
            if len(indices) > 0:
                row_reference = df.head(indices[-1] + 1).tail(1)
                features['steps_from_last_time_impression_appeared'].append(len(df) - indices[-1])
                features['time_elapsed_from_last_time_impression_appeared'].append(
                    int(clk['timestamp'].values[0] - row_reference['timestamp'].values[0]))
                features['kind_action_reference_appeared'].append(row_reference['action_type'].values[0])
            else:
                features['steps_from_last_time_impression_appeared'].append(-0.5)
                features['time_elapsed_from_last_time_impression_appeared'].append(-0.5)
                features['kind_action_reference_appeared'].append('no_action')
            features['times_impression_appeared'].append(len(indices))
            features['delta_position'].append(delta_pos)
            features['session_length'].append(session_length)
            features['session_steps'].append(session_steps)
            features['time_from_last_action'].append(time_from_last_action)
            features['reference_position_last_action'].append(reference_position_last_action)

            if submission_mode:
                features['label'].append(0)
            else:
                if int(clk['reference'].values[0]) == i:
                    features['label'].append(1)
                else:
                    features['label'].append(0)

            count += 1

        # zero padd the impressions with 0 feature values
        missing_impr_count = 25 - len(impr)
        if missing_impr_count > 0:
            for k in features.keys():
                if k == 'label':
                    features[k].extend(np.zeros(missing_impr_count))
                elif k == 'delta_position':
                    features[k].extend(np.ones(missing_impr_count) * 25)
                else:
                    features[k].extend(np.ones(missing_impr_count) * -1)
        return pd.DataFrame(features)


def _reinsert_clickout(df):
    # take the row of the missing clickout
    clickout_rows_df = df[(df['action_type'] == 'clickout item') & df['reference'].isnull()]
    # check if it exsists
    if len(clickout_rows_df)>0:
        # retrieve from the full_df the clickout
        missing_click = data.full_df().loc[clickout_rows_df.index[0]]['reference']
        # reinsert the clickout on the df
        df.at[clickout_rows_df.index[0], 'reference']= missing_click
    return df


def create_features_dataframe(mode, cluster):
    SAVE_PATH = f'dataset/preprocessed/FFNN_dataset/dataframes/{cluster}/{mode}'
    check_folder.check_folder(SAVE_PATH)

    # load TRAIN and TEST df
    train_df = data.train_df(mode, cluster)
    test_df = data.test_df(mode, cluster)

    print('extracting features from TRAIN...')
    train_features_dataframe = train_df.groupby(['user_id', 'session_id']).progress_apply(_extract_features)

    train_features_dataframe.to_csv(path_or_buf=f'{SAVE_PATH}/train_df.csv', index=False)
    del train_features_dataframe

    print('extracting features from TEST...')
    test_features_dataframe = test_df.groupby(['user_id', 'session_id']).progress_apply(_extract_features, submission_mode=True)
    test_features_dataframe.to_csv(path_or_buf=f'{SAVE_PATH}/test_df.csv', index=False)


def create_dataset_for_FFNN(mode, cluster, augmentation_power):
    SAVE_PATH = f'dataset/preprocessed/FFNN_dataset/{cluster}/{mode}'
    READ_PATH = f'dataset/preprocessed/FFNN_dataset/dataframes/{cluster}/{mode}/'
    check_folder.check_folder(SAVE_PATH)

    train_df = pd.read_csv(f'{READ_PATH}/train_df.csv')
    test_df = pd.read_csv(f'{READ_PATH}/test_df.csv')

    """
    CREATE DATA FOR TRAIN
    
    """
    X, Y = train_df.iloc[:, [0, 1, 2, 6, 7]], train_df.iloc[:, 5]
    X_session_features = train_df.iloc[:, [9, 10, 11, 12]]
    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()

    del train_df

    print('scaling...')
    # normalize the values
    X_session_features_norm = scaler1.fit_transform(X_session_features)
    X_norm = scaler2.fit_transform(X)
    Y_norm = Y.values

    # removing duplicates from session featureS
    X_session_features = []
    for i in range(0, X_session_features_norm.shape[0], 25):
        for j in range(augmentation_power):
            X_session_features.append(X_session_features_norm[i])
    X_session_features = np.array(X_session_features)

    # shuffle the data
    print('shuffling...')



    X_norm_shuffled = []
    Y_norm_shuffled = []
    for i in tqdm(range(0, X_norm.shape[0], 25)):
        for j in range(augmentation_power):
            x, y = shuffle(X_norm[i:i + 25], Y_norm[i:i + 25])
            X_norm_shuffled.append(x)
            Y_norm_shuffled.append(y)

    del X_norm
    del Y_norm

    print('reshaping...')
    # create the train and test data to be saved
    data_train = np.array(X_norm_shuffled).reshape((-1, 25 * 5))  # 25* NUM_FEATURES
    labels = np.array(Y_norm_shuffled)

    # add the session features to the samples
    data_train = np.concatenate((data_train, X_session_features), axis=1)
    print(data_train.shape)

    X_train, X_val, Y_train, Y_val = train_test_split(data_train, labels, shuffle=False, test_size=0.2)

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
    X = test_df.iloc[:, [0, 1, 2, 6, 7]]
    X_session_features = test_df.iloc[:, [9, 10, 11, 12]]
    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()

    # retrieving target indeces
    target_indices_duplicated = test_df.iloc[:, -1]
    target_indeces = []
    for i in range(0, target_indices_duplicated.shape[0], 25):
        target_indeces.append(target_indices_duplicated[i])

    target_indeces = np.array(target_indeces)

    del test_df

    print('scaling...')
    # normalize the values
    X_session_features_norm = scaler1.fit_transform(X_session_features)
    X_norm = scaler2.fit_transform(X)

    # removing duplicates from session featureS
    X_session_features = []
    for i in range(0, X_session_features_norm.shape[0], 25):
        X_session_features.append(X_session_features_norm[i])
    X_session_features = np.array(X_session_features)

    print('reshaping...')
    # create the train and test data to be saved
    data_train = np.array(X_norm).reshape((-1, 25 * 5))  # 25* NUM_FEATURES

    # add the session features to the samples
    data_train = np.concatenate((data_train, X_session_features), axis=1)

    print('saving training data...')
    np.save(f'{SAVE_PATH}/X_test', data_train)
    np.save(f'{SAVE_PATH}/target_indeces', target_indeces)
    print('done')



if __name__ == '__main__':
    #create_features_dataframe('full', 'no_cluster')
    #create_dataset_for_FFNN('small', 'no_cluster', augmentation_power=4)

    nn_dict_params = {
        'activation_function_internal_layers': 'relu',
        'neurons_per_layer': 150,
        'loss': 'categorical_crossentropy',
        'optimizer': 'adam',
        'validation_split': 0.2,
        'epochs': 100,
        'batch_size': 128,
    }

    model = NeuralNetworks(mode='small', cluster='no_cluster', nn_dict_params=nn_dict_params)
    model.evaluate()

    #out.create_sub(recs, 'prova')







