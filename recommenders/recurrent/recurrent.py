import sys
import os
sys.path.append(os.getcwd())

import data
import pandas as pd
import numpy as np
from numpy.linalg import norm as L2Norm
from recommenders.recommender_base import RecommenderBase
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import preprocess_utils.session2vec as sess2vec
from utils.check_folder import check_folder
import time
from tqdm import tqdm

class Recurrent_Recommender(RecommenderBase):
    
    def __init__(self, mode, cell_type, num_units, num_layers, embedding_size=None, validation_percentage=0.15):
        # Create the recurrent model
        # n_input_features: number of features of the input
        # num_units:        number of memory cells
        assert num_layers > 0
        assert num_units > 0
        assert mode in ['full', 'local', 'small']
        assert cell_type in ['LSTM', 'lstm', 'GRU', 'gru']

        self.name = cell_type.upper()
        self.mode = mode
        self.cluster = 'cluster_recurrent'

        # load the dataset of vectorized sessions
        print(f'Loading {mode} dataset...', end=' ', flush=True)
        X_train_df, Y_train_df, self.X_test_df, self.Y_test_df = sess2vec.load_and_prepare_dataset(mode)

        # groups sessions based on 'user_id' and 'session_id' to fast access their interactions indices
        train_session_groups_df = self._get_session_groups_indices_df(X_train_df)
        # shuffle the sessions
        train_session_groups_df = train_session_groups_df.sample(frac=1)
        train_count = train_session_groups_df.shape[0]
        validation_session_groups_df = train_session_groups_df.iloc[-int(train_count * validation_percentage):]
        train_session_groups_df = train_session_groups_df.iloc[0:int(train_count * (1-validation_percentage))]

        #test_session_groups_df = self._get_session_groups_indices_df(self.X_test_df)
        
        print('Done!')

        # number of groups to process during traininig and validation
        self.steps_per_epoch = train_session_groups_df.shape[0]
        self.steps_per_validation = validation_session_groups_df.shape[0]
        #self.test_steps = test_session_groups_df.shape[0]
        
        # build the generators
        self.train_generator = self._get_session_batch_generator(X_train_df, Y_train_df, train_session_groups_df)
        self.validation_generator = self._get_session_batch_generator(X_train_df, Y_train_df, validation_session_groups_df)
        #self.test_generator = self._get_prediction_session_batch_generator(X_test_df, test_session_groups_df)

        # input features are shape - 2 because we drop 'user_id' and 'session_id'
        input_features_count = X_train_df.shape[1]-2
        input_shape = (None, input_features_count)
        output_size = Y_train_df.shape[1]
        
        self.model = Sequential()
        CELL = LSTM if self.name == 'LSTM' else GRU

        self.has_embedding = isinstance(embedding_size, int) and embedding_size > 0
        if self.has_embedding:
            self.model.add( Embedding(input_features_count, embedding_size) )
            self.model.add( CELL(num_units, input_shape=(None, embedding_size), return_sequences=(num_layers>1)) )
            for i in range(num_layers-1):
                self.model.add( CELL(num_units, return_sequences=(i < num_layers-2)) )
        else:
            self.model.add( CELL(num_units, input_shape=input_shape, return_sequences=True) )
            for i in range(num_layers-1):
                self.model.add( CELL(num_units, return_sequences=True) )
        
        self.model.add( Dense(output_size, activation='sigmoid') )
        self.model.compile(sample_weight_mode='temporal', loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        print(self.model.summary())

    def _get_session_groups_indices_df(self, X_df, cols_to_group=['user_id','session_id'], indices_col_name='intrctns_indices'):
        """
        Return a dataframe with columns 'user_id', 'session_id', 'intrctns_indices'. This is used to retrieve the interaction indices
        from a particular session id of a user
        """
        return X_df.groupby(cols_to_group).apply(lambda r: pd.Series({indices_col_name: r.index.values})).reset_index()

    def _get_session_batch_generator(self, X, Y, sess_groups, indices_col_name='intrctns_indices'):
        """ Generator function for extracting a single session (1-batch) of the training set. """
        while True:
            for _, g in sess_groups.iterrows():
                intrctns_indices = g[indices_col_name]

                x_batch = X.drop(['user_id','session_id'], axis=1).loc[intrctns_indices]
                y_batch = Y.loc[intrctns_indices]
                # the weights are 1 for the clickout_item interaction and 0 for all the others
                weights = x_batch['action_type_clickout item'].values + 0.05
                #print(weights)

                if not self.has_embedding:
                    # x_batch, y_batch = np.expand_dims(x_batch, axis=0), np.expand_dims(y_batch, axis=0)
                    x_batch, y_batch, weights = np.expand_dims(x_batch, axis=0), np.expand_dims(y_batch, axis=0), np.expand_dims(weights, axis=0)
                # sample weight can be added as third element!!
                #yield x_batch, y_batch, sample_weight
                yield x_batch, y_batch, weights
    
    def _get_prediction_session_batch_generator(self, X, sess_groups, indices_col_name='intrctns_indices'):
        """ Generator function that extracts a single session (1-batch) of the test set. """
        while True:
            for _, g in sess_groups.iterrows():
                intrctns_indices = g[indices_col_name]

                x_batch = X.drop(['user_id','session_id'], axis=1).loc[intrctns_indices]

                if not self.has_embedding:
                    x_batch = np.expand_dims(x_batch, axis=0)
                #print(x_batch.shape)
                yield x_batch


    def fit(self, epochs, early_stopping_patience=10, checkpoints_path='recommenders/recurrent/checkpoints', tensorboard_path='recommenders/recurrent/tensorboard'):
        callbacks = []
        # early stopping callback
        if isinstance(early_stopping_patience, int):
            assert early_stopping_patience > 0
            callbacks.append( EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=1) )
        # checkpoint callback
        if isinstance(checkpoints_path, str):
            check_folder(checkpoints_path)
            datetime = time.strftime('%d-%m-%Y')
            filename = f'{self.name}_{datetime}_' + 'epoch{epoch:03d}.hdf5'
            chkp_path = os.path.join(checkpoints_path, filename)
            callbacks.append( ModelCheckpoint(filepath=chkp_path, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True) )
        # tensorboard callback
        if isinstance(tensorboard_path, str):
            check_folder(tensorboard_path)
            callbacks.append( TensorBoard(log_dir=tensorboard_path, histogram_freq=0, write_graph=False) )
        
        self.history = self.model.fit_generator(generator=self.train_generator, epochs=epochs, steps_per_epoch=self.steps_per_epoch,
                                                validation_data=self.validation_generator, validation_steps=self.steps_per_validation,
                                                callbacks=callbacks)

    def load_from_checkpoint(self, filename):
        self.model.load_weights(filename)

    def recommend_batch(self):
        # load the test dataframe
        #test_df = data.test_df(self.mode, cluster='cluster_recurrent').reset_index()
        
        #accomodations_one_hot = 
        test_session_groups_df = self._get_session_groups_indices_df(self.X_test_df, indices_col_name='intrctns_indices')
        
        target_indices = data.target_indices(self.mode, self.cluster)

        # predict the references
        predictions = []
        for _,row in test_session_groups_df.iterrows():
            x = self.X_test_df.loc[row.intrctns_indices].drop(['user_id','session_id'],axis=1)
            preds = self.model.predict(np.expand_dims(x,axis=0), steps=1)
            predictions.append(preds[0])
        
        predictions = np.concatenate(predictions)
        predictions_df = pd.DataFrame(predictions, columns=self.Y_test_df.columns)
        predictions_df.index = self.Y_test_df.index

        predictions = predictions_df.loc[target_indices].values
        del predictions_df

        full_df = data.full_df()
        dict_col = data.dictionary_col(self.mode, 'urm_recurrent_models', 'session')
        #icm = sim.normalization.bm25(data.icm().tocsr(), axis=1)
        icm = data.icm().tocsr()

        result_predictions = []

        assert len(predictions) == len(target_indices)
        
        for index in tqdm(target_indices):
            # get the impressions of the clickout to predict
            impr = list(map(int, full_df.loc[index]['impressions'].split('|')))
            # get the rows of the icm
            icm_rows = icm[[dict_col[i] for i in impr]]
            # build a list of (impression, l2norm)
            prediction_impressions_distances = [ (impr[j], L2Norm(icm_rows[j] - predictions[j])) for j in range(len(impr)) ]
            # order the list based on the l2norm
            prediction_impressions_distances.sort(key=lambda tup: tup[1], reverse=True)
            # get only the impressions ids
            ordered_impressions = list(map(lambda x: x[0], prediction_impressions_distances))
            # append the couple (index, reranked impressions)
            result_predictions.append( (index, ordered_impressions) )

        print('prediction created !!!')

        return result_predictions

    def get_scores_batch(self):
        pass
    
    def plot_info(self):
        if self.history is not None:
            plt.figure()
            plt.plot(self.history.history['loss'])
            plt.show()


if __name__ == "__main__":
    import utils.menu as menu

    mode = menu.mode_selection() 
    cell_type = menu.single_choice('Choose a cell mode:', ['LSTM', 'GRU', 'auto'], [lambda: 'LSTM', lambda: 'GRU', lambda: 'auto'])
    print()
    if cell_type == 'auto':
        cell_type = 'GRU'
        epochs = 1
        embed_size = -1
        layers = 1
        units = 4
        tb_path = None
    else:
        epochs = input('Insert number of epochs: ')

        embeddings = ['None',16, 32, 64, 128]
        embed_size = menu.single_choice('Do you want to add a first embedding layer?', embeddings, [lambda: -1, lambda: 16, lambda: 32, lambda: 64, lambda: 128])
        layers = input('Insert number of layers: ')
        units = input('Insert number of units per layer: ')
        tb_path = menu.yesno_choice('Do you want to enable Tensorboard?', lambda: 'recommenders/tensorboard', lambda: None)

    model = Recurrent_Recommender(mode=mode, cell_type=cell_type, num_layers=int(layers), num_units=int(units), embedding_size=embed_size)
    model.fit(epochs=int(epochs), tensorboard_path=tb_path)

    recommendations = model.recommend_batch()
    #print(recommendations)
    print('Recommendation count: ', len(recommendations))
    
    #model.evaluate()
    model.compute_MRR(recommendations)
