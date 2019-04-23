import sys
import os
sys.path.append(os.getcwd())

import data
import pandas as pd
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from recommenders.recommender_base import RecommenderBase
import preprocess_utils.session2vec as sess2vec
from utils.dataset import SequenceDataset
from numpy.linalg import norm as L2Norm

from utils.check_folder import check_folder
import matplotlib.pyplot as plt
from tqdm import tqdm


class RecurrentRecommender(RecommenderBase):
    
    def __init__(self, dataset, cell_type, num_units, num_layers, use_generator=True, validation_split=0.15,
                 loss='mean_squared_error', optimizer='rmsprop', checkpoints_path=None, tensorboard_path=None):
        """ Create the recurrent model
        dataset (Dataset): dataset object to use
        num_units:        number of memory cells
        
        """
        assert num_layers > 0
        assert num_units > 0
        assert cell_type in ['LSTM', 'lstm', 'GRU', 'gru']

        self.dataset = dataset
        super().__init__(dataset.mode, dataset.cluster, 'RecurrentRecommender_{}'.format(cell_type.upper()))

        self.validation_split = validation_split
        self.use_generator = use_generator
        self.checkpoints_path = checkpoints_path
        self.tensorboard_path = tensorboard_path
        
        if use_generator:
            # generator
            self.train_gen, self.val_gen = dataset.get_train_validation_generator(validation_split)
            batch_x, batch_y, _ = self.train_gen.__getitem__(0)
            input_shape = (None, batch_x.shape[1], batch_x.shape[2])
            output_size = batch_y.shape[-1]
        else:
            # full dataset
            self.X, self.Y = dataset.load_Xtrain(), dataset.load_Ytrain()
            input_shape = self.X.shape
            output_size = self.Y.shape[-1]
        
        # build the model
        
        CELL = LSTM if self.name == 'LSTM' else GRU
        self.model = Sequential()
        self.model.add( CELL(num_units, input_shape=(input_shape[1], input_shape[2]), dropout=0.1, return_sequences=True) )
        for i in range(num_layers-1):
            self.model.add( CELL(num_units, dropout=0.1, return_sequences=(i < num_layers-1) ))

        self.model.add( Dense(output_size, activation='sigmoid') )
        self.model.add( Dense(output_size, activation='sigmoid') )
        self.model.compile(sample_weight_mode='temporal', loss=loss, optimizer=optimizer, metrics=['accuracy'])

        print(self.model.summary())


    def fit(self, epochs, early_stopping_patience=10):
        callbacks = []
        # early stopping callback
        if isinstance(early_stopping_patience, int):
            assert early_stopping_patience > 0
            callbacks.append( EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=1) )
        
        # tensorboard callback
        if isinstance(self.tensorboard_path, str):
            check_folder(self.tensorboard_path)
            callbacks.append( TensorBoard(log_dir=self.tensorboard_path, histogram_freq=0, write_graph=False) )
        
        if self.use_generator:
            self.history = self.model.fit_generator(self.train_gen, epochs=epochs, validation_data=self.val_gen, callbacks=callbacks)
        else:
            self.history = self.model.fit(self.X, self.Y, epochs=epochs, validation_split=self.validation_split, callbacks=callbacks)
        

    def load_from_checkpoint(self, filename):
        self.model.load_weights(filename)

    def recommend_batch(self):
        X, indices = self.dataset.load_Xtest()
        
        # predict the references
        predictions = self.model.predict(X)
        
        # flatten X and the indices to be 2-dimensional
        predictions = predictions.reshape((-1, predictions.shape[-1]))
        indices = indices.flatten()
        
        # take only the target predictions
        pred_df = pd.DataFrame(predictions)
        pred_df['orig_index'] = indices
        pred_df = pred_df.set_index('orig_index')
        predictions = pred_df.loc[target_indices].sort_index().values
        del pred_df
        
        #test_session_groups_df = self._get_session_groups_indices_df(self.X_test_df, indices_col_name='intrctns_indices')
        
        #predictions = []
        #for _,row in test_session_groups_df.iterrows():
        #    x = self.X_test_df.loc[row.intrctns_indices].drop(['user_id','session_id'],axis=1)
        #    preds = self.model.predict(np.expand_dims(x,axis=0), steps=1)
        #    predictions.append(preds[0])
        #predictions = np.concatenate(predictions)        
        
        #predictions_df = pd.DataFrame(predictions, columns=self.Y_test_df.columns)
        #predictions_df.index = self.Y_test_df.index
        #predictions = predictions_df.loc[target_indices].values
        #del predictions_df

        full_df = data.train_df('full')
        accomodations1hot_df = data.accomodations_one_hot()

        result_predictions = []

        assert len(predictions) == len(target_indices)
        
        for k,index in tqdm(enumerate(target_indices)):
            # get the impressions of the clickout to predict
            impr = list(map(int, full_df.loc[index]['impressions'].split('|')))
            # get the true labels from the accomodations one-hot
            true_labels = accomodations1hot_df.loc[impr].values
            # build a list of (impression, l2norm distance)
            prediction_impressions_distances = [ (impr[j], L2Norm(true_labels[j] - predictions[k])) for j in range(len(impr)) ]
            # order the list based on the l2norm (smaller distance is better)
            prediction_impressions_distances.sort(key=lambda tup: tup[1])
            # get only the impressions ids
            ordered_impressions = list(map(lambda x: x[0], prediction_impressions_distances))
            # append the couple (index, reranked impressions)
            result_predictions.append( (index, ordered_impressions) )

        print('prediction created !!!')

        return result_predictions

    def plot_info(self):
        if self.history is not None:
            plt.figure()
            plt.plot(self.history.history['loss'])
            plt.show()
            
    def get_scores_batch(self):
        return None


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
        epochs = int(input('Insert number of epochs: '))
        layers = int(input('Insert number of layers: '))
        units = int(input('Insert number of units per layer: '))
        tb_path = menu.yesno_choice('Do you want to enable Tensorboard?', lambda: 'recommenders/tensorboard', lambda: None)

    dataset = SequenceDataset(f'dataset/preprocessed/cluster_recurrent/{mode}')
    
    # model = RecurrentRecommender(dataset, use_generator=True, cell_type='gru', num_units=50, num_layers=2)
    # model.fit(epochs=5)
    model = RecurrentRecommender(dataset, use_generator=True, cell_type=cell_type, num_layers=layers, num_units=units)
    model.fit(epochs=epochs)

    print()
    print('Fit completed!')

    target_indices = data.target_indices(mode, 'cluster_recurrent')
    
    recommendations = model.recommend_batch()
    #print(recommendations)
    print('Recommendation count: ', len(recommendations))
    
    model.compute_MRR(recommendations)
