import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import preprocess_utils.session2vec as sess2vec
from utils.check_folder import check_folder

class GRU_Recommender:
    
    def __init__(self, mode, num_units, num_layers):
        # Create the recurrent model
        # n_input_features: number of features of the input
        # num_units:        number of memory cells
        assert num_layers > 0

        # load the dataset of vectorized sessions
        X_train_df, Y_train_df, X_test_df, Y_test_df = sess2vec.load_and_prepare_dataset(mode)

        # groups sessions based on 'user_id' and 'session_id' to fast access their interactions indices
        train_session_groups = self._get_session_groups_indices_df(X_train_df, Y_train_df)
        validation_session_groups = self._get_session_groups_indices_df(X_test_df, Y_test_df)

        # number of groups to process during traininig and validation
        self.steps_per_epoch = train_session_groups.shape[0]
        self.steps_per_validation = validation_session_groups.shape[0]
        
        # build the generators
        self.train_generator = self._get_session_batch_generator(X_train_df, Y_train_df, train_session_groups)
        self.validation_generator = self._get_session_batch_generator(X_test_df, Y_test_df, validation_session_groups)

        # input features are shape - 2 because we drop 'user_id' and 'session_id'
        input_shape = (None, X_train_df.shape[1]-2)
        output_size = Y_train_df.shape[1]
        
        self.model = Sequential()
        self.model.add(GRU(num_units, input_shape=input_shape, return_sequences=True))
        for _ in range(num_layers-1):
            self.model.add(GRU(num_units, return_sequences=True))
        self.model.add(Dense(output_size, activation='sigmoid'))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        print(self.model.summary())

    def _get_session_groups_indices_df(self, X_df, Y_df, cols_to_group=['user_id','session_id'], indices_col_name='intrctns_indices'):
        """
        Return a dataframe with columns 'user_id', 'session_id', 'intrctns_indices'. This is used to retrieve the interaction indices
        from a particular session id of a user
        """
        return X_df.groupby(cols_to_group).apply(lambda r: pd.Series({indices_col_name: r.index.values})).reset_index()

    def _get_session_batch_generator(self, X, Y, sess_groups, indices_col_name='intrctns_indices'):
        """ Generator function for extracting a single random session (1-batch) of the training set. """
        while True:
            for _, g in sess_groups.iterrows():
                intrctns_indices = g[indices_col_name]

                x_batch = X.drop(['user_id','session_id'], axis=1).loc[intrctns_indices]
                y_batch = Y.loc[intrctns_indices]

                yield np.expand_dims(x_batch, axis=0), np.expand_dims(y_batch, axis=0)


    def fit_generator(self, epochs, early_stopping_patience=10, checkpoints_path='recommenders/checkpoints', tensorboard_path='recommenders/tensorboard'):
        callbacks = []
        # early stopping callback
        if isinstance(early_stopping_patience, int):
            assert early_stopping_patience > 0
            callbacks.append( EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=1) )
        # checkpoint callback
        if isinstance(checkpoints_path, str):
            check_folder(checkpoints_path)
            callbacks.append( ModelCheckpoint(filepath=checkpoints_path, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True) )
        # tensorboard callback
        if isinstance(tensorboard_path, str):
            check_folder(tensorboard_path)
            callbacks.append( TensorBoard(log_dir=tensorboard_path, histogram_freq=0, write_graph=False) )
        
        self.history = self.model.fit_generator(generator=self.train_generator, epochs=epochs, steps_per_epoch=self.steps_per_epoch,
                                                validation_data=self.validation_generator, validation_steps=self.steps_per_validation,
                                                callbacks=callbacks)

    def predict(self, X):
        return self.model.predict(X)

    def plot_info(self):
        if self.history is not None:
            plt.figure()
            plt.plot(self.history.history['loss'])
            plt.show()


if __name__ == "__main__":
    import utils.menu as menu

    mode = input('Insert dataset mode: ')
    epochs = input('Insert number of epochs: ')
    units = input('Insert number of units: ')
    tb_path = menu.yesno_choice('Do you want to enable Tensorboard?', lambda: 'recommenders/tensorboard', lambda: None)

    model = GRU_Recommender(mode, num_layers=1, num_units=int(units))
    model.fit_generator(epochs=int(epochs), tensorboard_path=tb_path)

    