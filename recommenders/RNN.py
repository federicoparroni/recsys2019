import os
import data
import pandas as pd
import numpy as np
import time
from abc import abstractmethod

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, GRU, TimeDistributed, BatchNormalization, Activation, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import metrics

from recommenders.recommender_base import RecommenderBase
import preprocess_utils.session2vec as sess2vec
from sklearn.utils import shuffle
from numpy.linalg import norm as L2Norm

from utils.check_folder import check_folder
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.telegram_bot import TelegramBotKerasCallback


def mrr(y_true, y_pred):
    y_true = y_true
    y_pred = y_pred
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



class RecurrentRecommender(RecommenderBase):
    
    def __init__(self, dataset, input_shape, cell_type, num_recurrent_layers, num_recurrent_units, num_dense_layers, output_size,
                use_generator=False, validation_split=0.15, use_batch_normalization=False, bidirectional=False,
                loss='mean_squared_error', optimizer='rmsprop', class_weights=None, sample_weights=None,
                metrics=['accuracy', mrr], batch_size=64, checkpoints_path=None, tensorboard_path=None):
        """ Create the recurrent model
        dataset (Dataset):          dataset to use
        input_shape (int):          shape of the input samples (ex: (6,10) for session with length 6 and 10 features)
        cell_type (str):            recurrent cell type (LSTM, GRU)
        num_recurrent_layers (int): number of recurrent layers (> 0)
        num_recurrent_units (int):  number of recurrent cells (> 0)
        num_dense_layers (int):     number of dense layer before the output (> 0)
        output_size (int):          size of the ouput vector (> 0)
        use_generator (bool):       whether to use a batch generator or the entire dataset to train the model
        validation_split (float):   percentage of the training samples to use as validation set
        use_batch_normalization (bool): whether to use batch normalization after each dense layer
        loss (str):                 loss to minimize
        optimizer (str):            optimizer to use to minimize the loss
        class_weights (dict):       values to weight each class with (keys are classes, values are weights)
        weigth_samples (bool):      whether to weight only the last clickout with the class weights or not
        """
        assert num_recurrent_layers > 0
        assert num_recurrent_units > 0
        assert num_dense_layers > 0
        assert cell_type in ['LSTM', 'lstm', 'GRU', 'gru']

        self.dataset = dataset
        self.validation_split = validation_split
        self.class_weights = class_weights
        self.sample_weights = sample_weights
        self.use_weights = self.class_weights is not None

        assert len(input_shape) == 2
        self.input_shape = input_shape

        self.metrics = metrics
        self.use_generator = use_generator
        self.batch_size = batch_size
        self.checkpoints_path = checkpoints_path
        self.tensorboard_path = tensorboard_path

        name = 'rnn_{}_{}layers_{}units_{}dense'.format(cell_type.upper(), num_recurrent_layers, num_recurrent_units, num_dense_layers)
        name += '_wgt' if self.use_weights else ''
        super(RecurrentRecommender, self).__init__(dataset.mode, dataset.cluster, name=name)
        
        if use_generator:
            # generator
            self.test_gen = dataset.get_test_generator()
        
        # build the model
        self.build_model(input_shape=input_shape, cell_type=cell_type, num_recurrent_layers=num_recurrent_layers,
                            num_recurrent_units=num_recurrent_units, num_dense_layers=num_dense_layers, bidirectional=bidirectional,
                            output_size=output_size, use_batch_normalization=use_batch_normalization)
        
        #if self.weight_samples:
        #    self.model.compile(sample_weight_mode='temporal', loss=loss, optimizer=optimizer, metrics=self.metrics)
        #else:
        self.model.compile(loss=loss, optimizer=optimizer, metrics=self.metrics)

        print(self.model.summary())
        print()
    

    def build_model(self, input_shape, cell_type, num_recurrent_layers, num_recurrent_units, num_dense_layers,
                    bidirectional, output_size, use_batch_normalization):
        CELL = LSTM if cell_type == 'LSTM' else GRU
        self.model = Sequential()

        #self.model.add( TimeDistributed(Dense(num_recurrent_units, activation='relu'), input_shape=self.input_shape) )

        if bidirectional:
            self.model.add( Bidirectional(CELL(num_recurrent_units, dropout=0.2, recurrent_dropout=0.2,
                                    return_sequences=(num_recurrent_layers > 1) ), input_shape=self.input_shape))
        else:
            self.model.add( CELL(num_recurrent_units, dropout=0.2, recurrent_dropout=0.2,
                                    return_sequences=(num_recurrent_layers > 1), input_shape=self.input_shape))
        for i in range(num_recurrent_layers-1):
            if bidirectional:
                self.model.add( Bidirectional(CELL(num_recurrent_units, dropout=0.2, recurrent_dropout=0.2,
                                                    return_sequences=(i < num_recurrent_layers-2) )))
            else:
                self.model.add( CELL(num_recurrent_units, dropout=0.2, recurrent_dropout=0.2,
                                        return_sequences=(i < num_recurrent_layers-2) ))

        # time distributed
        #self.model.add( TimeDistributed(Dense(num_recurrent_units, activation='relu')) )

        if num_dense_layers > 1:
            dense_neurons = np.linspace(num_recurrent_units, output_size, num_dense_layers)
            for n in dense_neurons[:-1]:
                if use_batch_normalization:
                    self.model.add( Dense(int(n), activation=None) )
                    self.model.add( BatchNormalization() )
                    self.model.add( Activation('relu') )
                else:
                    self.model.add( Dense(int(n), activation='relu') )
                self.model.add( Dropout(rate=0.1) )
        
        # add the last dense layer
        if use_batch_normalization:
            self.model.add( Dense(output_size, activation=None) )
            self.model.add( BatchNormalization() )
            self.model.add( Activation('softmax') )
        else:
            self.model.add( Dense(output_size, activation='softmax') )
        self.model.add( Dropout(rate=0.1) )
        

    def fit(self, epochs, early_stopping_patience=10, early_stopping_on='val_loss', mode='min'):
        #weights = self.class_weights if self.weight_samples else []

        callbacks = [ TelegramBotKerasCallback(log_every_epochs=1, account='parro') ]
        # early stopping callback
        if isinstance(early_stopping_patience, int):
            assert early_stopping_patience > 0
            callbacks.append( EarlyStopping(monitor=early_stopping_on, patience=early_stopping_patience, mode=mode,
                                            verbose=1, restore_best_weights=True) )
        
        # tensorboard callback
        if isinstance(self.tensorboard_path, str):
            check_folder(self.tensorboard_path)
            callbacks.append( TensorBoard(log_dir=self.tensorboard_path, histogram_freq=0, write_graph=False) )
        
        if self.use_generator:
            self.train_gen, self.val_gen = self.dataset.get_train_validation_generator(self.validation_split) #, weights)
            assert self.train_gen.__getitem__(0)[0].shape[1:] == self.input_shape

            self.history = self.model.fit_generator(self.train_gen, epochs=epochs, validation_data=self.val_gen,
                                                    callbacks=callbacks, max_queue_size=3, class_weight=self.class_weights)
        else:
            self.X, self.Y = self.dataset.load_Xtrain(), self.dataset.load_Ytrain()
            self.X, self.Y = shuffle(self.X, self.Y)
            
            self.history = self.model.fit(self.X, self.Y, epochs=epochs, batch_size=self.batch_size,
                                            validation_split=self.validation_split, callbacks=callbacks,
                                            class_weight=self.class_weights, sample_weight=self.sample_weights)

    def fit_cv(self, x, y, groups, train_indices, test_indices, epochs, early_stopping_patience=None, early_stopping_on='val_loss', mode='min'):
        callbacks = [ TelegramBotKerasCallback(log_every_epochs=1, account='parro') ]
        # early stopping callback
        if isinstance(early_stopping_patience, int):
            assert early_stopping_patience > 0
            callbacks.append( EarlyStopping(monitor=early_stopping_on, patience=early_stopping_patience, mode=mode,
                                            verbose=1, restore_best_weights=True) )
        
        # fit on the data, dropping the index
        cw = None if self.class_weights is None else self.class_weights[train_indices]
        sw = None if self.sample_weights is None else self.sample_weights[train_indices]
        
        self.model.fit(x[train_indices,:,1:], y[train_indices], epochs=epochs, batch_size=self.batch_size,
                        #validation_data=(x_val[:,:,1:], y_val),
                        callbacks=callbacks, class_weight=cw, sample_weight=sw)

    def save(self, folderpath, suffix=''):
        """ Save the full state of the model, including:
        - the architecture
        - the weights
        - the training configuration (loss, optimizer)
        - the state of the optimizer (allowing to resume the training)
        See: https://keras.io/getting-started/faq/#savingloading-whole-models-architecture-weights-optimizer-state
        """
        check_folder(folderpath)
        path = os.path.join(folderpath, '{}{}.h5'.format(self.name, suffix))
        self.model.save(path)
    
    def load(self, path):
        """ Load the full state of a model """
        self.model = load_model(path, custom_objects={"mrr": mrr})
        # reset the name to the one of the checkpoint
        self.name = ''.join(os.path.basename(path).split('.')[0:-1])


    def load_from_checkpoint(self, filename):
        self.model.load_weights(filename)

    def plot_info(self):
        if self.history is not None:
            plt.figure()
            plt.plot(self.history.history['loss'])
            plt.show()

    @abstractmethod
    def recommend_batch(self):
        pass

    @abstractmethod
    def get_scores_batch(self):
        pass

