import os
import data
import pandas as pd
import numpy as np
import time
from abc import abstractmethod

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, GRU, Embedding, BatchNormalization, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import metrics

from recommenders.recommender_base import RecommenderBase
import preprocess_utils.session2vec as sess2vec
from numpy.linalg import norm as L2Norm

from utils.check_folder import check_folder
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.telegram_bot import TelegramBotKerasCallback


class RecurrentRecommender(RecommenderBase):
    
    def __init__(self, dataset, cell_type, num_recurrent_layers, num_recurrent_units, num_dense_layers, output_size,
                use_generator=True, validation_split=0.15, use_batch_normalization=False,
                loss='mean_squared_error', optimizer='rmsprop', class_weights=[],
                checkpoints_path=None, tensorboard_path=None):
        """ Create the recurrent model
        dataset (Dataset): dataset object to use
        num_recurrent_units (int): number of memory cells
        
        """
        assert num_recurrent_layers > 0
        assert num_recurrent_units > 0
        assert num_dense_layers > 0
        assert cell_type in ['LSTM', 'lstm', 'GRU', 'gru']

        self.dataset = dataset
        self.validation_split = validation_split
        self.class_weights = np.array(class_weights)
        self.use_weights = len(self.class_weights) > 0
        self.use_generator = use_generator
        self.checkpoints_path = checkpoints_path
        self.tensorboard_path = tensorboard_path

        name = 'recurrent_{}_{}layers_{}units'.format(cell_type.upper(), num_recurrent_layers, num_recurrent_units)
        name += '_w' if self.use_weights else ''
        super().__init__(dataset.mode, dataset.cluster, name=name)
        
        if use_generator:
            # generator
            self.test_gen = dataset.get_test_generator()
            if self.use_weights:
                batch_x = self.test_gen.__getitem__(0)
            else:
                batch_x = self.test_gen.__getitem__(0)
            input_shape = (None, batch_x.shape[1], batch_x.shape[2])
            #output_size = batch_y.shape[-1]
        else:
            # full dataset
            self.X, self.Y = dataset.load_Xtrain(), dataset.load_Ytrain()
            input_shape = self.X.shape
            #output_size = self.Y.shape[-1]
        
        # build the model
        CELL = LSTM if self.name == 'LSTM' else GRU
        self.model = Sequential()
        self.model.add( CELL(num_recurrent_units, input_shape=(input_shape[1], input_shape[2]), dropout=0.2,recurrent_dropout=0.2, return_sequences=True) )
        for i in range(num_recurrent_layers-1):
            self.model.add( CELL(num_recurrent_units, dropout=0.2,recurrent_dropout=0.2, return_sequences=(i < num_recurrent_layers-1) ))

        if num_dense_layers > 1:
            dense_neurons = np.linspace(num_recurrent_units, output_size, num_dense_layers)
            for n in dense_neurons[:-1]:
                if use_batch_normalization:
                    self.model.add( Dense(int(n), activation=None) )
                    self.model.add( BatchNormalization() )
                    self.model.add( Activation('relu') )
                else:
                    self.model.add( Dense(int(n), activation='relu') )
        
        if use_batch_normalization:
            self.model.add( Dense(output_size, activation=None) )
            self.model.add( BatchNormalization() )
            self.model.add( Activation('softmax') )
        else:
            self.model.add( Dense(output_size, activation='softmax') )
        self.model.add( Dropout(rate=0.3) )
        
        self.model.compile(sample_weight_mode='temporal', loss=loss, optimizer=optimizer, metrics=['accuracy', self.mrr])

        print(self.model.summary())
        print()
        if self.use_generator:
            print('Train with batches of shape X: {}'.format(batch_x.shape))
        else:
            print('Train with a dataset of shape X: {} - Y: {}'.format(self.X.shape, self.Y.shape))

    def mrr(self, y_true, y_pred):
        y_true = y_true[:,-1,:]
        y_pred = y_pred[:,-1,:]
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

    def fit(self, epochs, early_stopping_patience=10):
        self.train_gen, self.val_gen = self.dataset.get_train_validation_generator(self.validation_split, class_weights=self.class_weights)

        callbacks = [ TelegramBotKerasCallback() ]
        # early stopping callback
        if isinstance(early_stopping_patience, int):
            assert early_stopping_patience > 0
            callbacks.append( EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=1) )
        
        # tensorboard callback
        if isinstance(self.tensorboard_path, str):
            check_folder(self.tensorboard_path)
            callbacks.append( TensorBoard(log_dir=self.tensorboard_path, histogram_freq=0, write_graph=False) )
        
        if self.use_generator:
            self.history = self.model.fit_generator(self.train_gen, epochs=epochs, validation_data=self.val_gen,
                                                    callbacks=callbacks, max_queue_size=3)
        else:
            self.history = self.model.fit(self.X, self.Y, epochs=epochs, validation_split=self.validation_split, 
                                            callbacks=callbacks)
        
    def save(self, folderpath):
        """ Save the full state of the model, including:
        - the architecture
        - the weights
        - the training configuration (loss, optimizer)
        - the state of the optimizer (allowing to resume the training)
        See: https://keras.io/getting-started/faq/#savingloading-whole-models-architecture-weights-optimizer-state
        """
        path = os.path.join(folderpath, '{}.h5'.format(self.name))
        self.model.save(path)
    
    def load(self, path):
        """ Load the full state of a model """
        self.model = load_model(path, custom_objects={"mrr": self.mrr})

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

