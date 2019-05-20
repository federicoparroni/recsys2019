import os
import sys
sys.path.append(os.getcwd())

import data
from utils.dataset import SequenceDatasetForBinaryClassification

import preprocess_utils.session2vec as sess2vec

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import datetime

from recommenders.recurrent.RNNBinaryClassificator import RNNBinaryClassificator

from numpy.linalg import norm as L2Norm
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, GRU, Embedding, Dropout, TimeDistributed
from keras.callbacks import EarlyStopping

if __name__ == "__main__":
    import utils.menu as menu
    tqdm.pandas()

    mode = menu.mode_selection()
    opt = menu.single_choice('Optimizer?', ['Adam','RMSProp'], ['adam','rmsprop'])
    lr = menu.single_choice('Learning rate?', ['e-3', 'e-4', 'e-5'], [1e-3, 1e-4, 1e-5])

    pad = menu.single_choice('Which dataset?', ['Padded 6','Padded 12'], [lambda: 6, lambda: 12])
    dataset = SequenceDatasetForBinaryClassification(f'dataset/preprocessed/cluster_recurrent/{mode}/dataset_binary_classification_p{pad}')

    print('Loading data...')
    x, y = dataset.load_Xtrain(), dataset.load_Ytrain()
    x, y = shuffle(x, y)
    print()

    perc = np.sum(y) / len(y)
    print('Class-1 perc: {}'.format(perc))

    weights = dataset.get_class_weights()
    
    # model
    m = Sequential()
    #m.add( TimeDistributed(Dense(64), input_shape=(6,68)) )
    m.add( GRU(64, input_shape=(6,68), recurrent_dropout=0.2, dropout=0.2, return_sequences=True) )
    m.add( GRU(32, recurrent_dropout=0.2, dropout=0.2, return_sequences=False) )
    m.add( Dense(32, activation='relu') )
    m.add( Dropout(0.2) )
    m.add( Dense(1, activation='sigmoid') )
    m.add( Dropout(0.1) )

    #sgd = keras.optimizers.SGD(lr=1e-6, decay=1e-6, momentum=0.9, nesterov=True)
    if opt == 'adam':
        optim = keras.optimizers.Adam(lr=lr)
    else:
        optim = keras.optimizers.RMSprop(lr=lr)
    m.compile(optim, loss='binary_crossentropy', metrics=['accuracy'])
    m.summary()

    # train
    m.fit(x=x, y=y, epochs=1000, validation_split=0.15, batch_size=64, class_weight=weights,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

    timenow = datetime.datetime.now()
    m.save('gru_binary_{}.h5'.format(timenow))

    # evaluate
    xtest, indices = dataset.load_Xtest()
    y_true = sess2vec.add_reference_binary_labels(data.full_df().loc[indices], actiontype_col='action_type', action_equals='clickout item').ref_class

    y_pred = m.predict_classes(xtest)

    print('Opt: {}'.format(opt))
    print('Lr: {}'.format(lr))
    print(classification_report(y_true, y_pred, target_names=['class1','class0']))

