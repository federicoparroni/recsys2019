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


class NeuralNetworks(RecommenderBase):

    def __init__(self, mode, cluster, nn_dict_params):
        name = 'NeuralNetwork'
        super(NeuralNetworks, self).__init__(mode=mode, cluster=cluster, name=name)

        self.nn_dict_params = nn_dict_params
        base_path_dataset = 'dataset/preprocessed/FFNN_dataset'
        self.X = np.load(f'{base_path_dataset}/X.npy')
        self.Y = np.load(f'{base_path_dataset}/Y.npy')
        self._create_model()


    def fit(self):
        callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None,
                                 restore_best_weights=True)
        self.model.fit(self.X, self.Y,
                       validation_split=self.nn_dict_params['validation_split'],
                       epochs=self.nn_dict_params['epochs'],
                       batch_size=self.nn_dict_params['batch_size'])

    def recommend_batch(self):
        #TODO
        # must be given another df
        if not self.has_fit:
            print('first fit the model !')
            exit(0)

        print('taking the predictions')
        predictions = self.model.predict(self.features_test_df)

        target_indices_duplicated = self.features_test_df.iloc[:, -1]
        target_indeces = []
        for i in range(0, target_indices_duplicated.shape[0], 25):
            target_indeces.append(target_indices_duplicated[i])

        final_predictions = []

        count = 0
        for index in tqdm(target_indeces):
            impr = list(map(int, data.full_df().loc[index]['impressions'].split('|')))
            max_index = predictions[count].argmax()
            if max_index < len(impr):
                #print(max_index)
                #print(impr[max_index])
                t = impr[0]
                impr[0] = impr[max_index]
                impr[max_index] = t
            final_predictions.append((index, impr))
            count += 1

        return final_predictions

    def get_scores_batch(self):
        pass

    def _create_model(self):
        model = Sequential()
        model.add(Dense(self.nn_dict_params['neurons_per_layer'], input_dim=self.X.shape[1],
                        activation=self.nn_dict_params['activation_function_internal_layers']))
        #model.add(Dropout(rate=0.1))
        model.add(Dense(self.nn_dict_params['neurons_per_layer'],
                        activation=self.nn_dict_params['activation_function_internal_layers']))
        #model.add(Dropout(rate=0.1))
        model.add(Dense(self.nn_dict_params['neurons_per_layer'],
                        activation=self.nn_dict_params['activation_function_internal_layers']))
        model.add(Dense(25, activation='sigmoid'))

        # compile the model
        model.compile(loss=self.nn_dict_params['loss'], optimizer=self.nn_dict_params['optimizer'], metrics=[metrics.categorical_accuracy])
        self.model = model
        print('model created')


def _extract_features(df):
    # get the rows where the action is 'clickout item'

    clickout_rows_df = df[df['action_type'] == 'clickout item']

    if len(clickout_rows_df) > 0:

        # features
        features = {'times_impression_appeared': [],
                    'time_elapsed_from_last_time_impression_appeared': [],
                    'steps_from_last_time_impression_appeared': [],
                    'kind_action_reference_appeared': [],
                    'impression_position': [],
                    'label': [],
                    'price': [],
                    'price_position': [],
                    'reference_num': []}

        clk = clickout_rows_df.tail(1)
        head_index = df.head(1).index

        # considering only the past!
        # mantain of the df only the actions before the last clickout
        df = df.loc[head_index.values[0]:clk.index.values[0] - 1]

        # get the impression
        impr = clk['impressions'].values[0].split('|')

        prices = list(map(int, clk['prices'].values[0].split('|')))
        sorted_prices = prices.copy()
        sorted_prices.sort()

        references = df['reference'].values

        count = 0
        for i in impr:
            indices = np.where(references == str(i))[0]

            features['reference_num'].append(i)
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
                features['steps_from_last_time_impression_appeared'].append(0)
                features['time_elapsed_from_last_time_impression_appeared'].append(-1)
                features['kind_action_reference_appeared'].append('no_action')
            features['times_impression_appeared'].append(len(indices))

            if clk['reference'].values[0] == i:
                features['label'].append(1)
            else:
                features['label'].append(0)
            count += 1

        # zero padd the impressions with 0 feature values
        missing_impr_count = 25 - len(impr)
        if missing_impr_count > 0:
            for k in features.keys():
                features[k].extend(np.zeros(missing_impr_count))
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


def create_dataset_for_FFNN(mode, cluster):

    SAVE_PATH = 'dataset/preprocessed/FFNN_dataset'
    check_folder.check_folder(SAVE_PATH)

    # load TRAIN and TEST df
    train_df = data.train_df(mode, cluster)
    test_df = data.test_df(mode, cluster)

    print('reinserting clicks...')
    # reinsert the clickout on the test_df
    test_df_reconstructed = test_df.groupby(['user_id', 'session_id']).progress_apply(_reinsert_clickout)

    del test_df

    print('concat the two df...')
    # extract teh feature from both train and test df
    train_test_df = pd.concat([train_df, test_df_reconstructed])

    del train_df
    del test_df_reconstructed

    print('extracting features...')
    impressions_features_df = train_test_df.groupby(['user_id', 'session_id']).progress_apply(_extract_features)

    # divide label and features and normalize them
    # the 5 column is the label
    print('dividing data and labels...')
    X, Y = impressions_features_df.iloc[:, [0, 1, 2, 4, 6, 7]], impressions_features_df.iloc[:, 5]

    del impressions_features_df

    print('scaling...')
    scaler = MinMaxScaler()
    # normalize the values
    X_norm = scaler.fit_transform(X)
    Y_norm = Y.values

    # shuffle the data
    print('shuffling...')
    X_norm_shuffled = []
    Y_norm_shuffled = []
    for i in tqdm(range(0, X_norm.shape[0], 25)):
        x, y = shuffle(X_norm[i:i + 25], Y_norm[i:i + 25])
        X_norm_shuffled.append(x)
        Y_norm_shuffled.append(y)

    del X_norm
    del Y_norm

    print('reshaping...')
    # create the train and test data to be saved
    data_train = np.array(X_norm_shuffled).reshape((-1, 25 * 6))  # 25* NUM_FEATURES
    labels = np.array(Y_norm_shuffled)

    print('saving training data...')
    np.save(f'{SAVE_PATH}/X', data_train)
    print('done')

    print('saving labels...')
    np.save(f'{SAVE_PATH}/Y', labels)
    print('done')



if __name__ == '__main__':
    #create_dataset_for_FFNN('small', 'no_cluster')
    nn_dict_params = {
        'activation_function_internal_layers': 'relu',
        'neurons_per_layer': 300,
        'loss': 'categorical_crossentropy',
        'optimizer': 'rmsprop',
        'validation_split': 0.2,
        'epochs': 1000,
        'batch_size': 250,
    }
    model = NeuralNetworks(mode='small', cluster='no_cluster', nn_dict_params=nn_dict_params)
    model.fit()







