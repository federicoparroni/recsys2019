import sys
import os
sys.path.append(os.getcwd())

import data
import pandas as pd
import numpy as np
import keras

from recommenders.recurrent.RNNClassificationRecommender import RNNClassificationRecommender

from keras.layers import Dense, LSTM, GRU, Embedding, Dropout, TimeDistributed

import datetime
from keras.models import Sequential
from utils.dataset import SequenceDatasetForBinaryClassification
from sklearn.metrics import classification_report

from utils.check_folder import check_folder
import utils.telegram_bot as bot
from tqdm.auto import tqdm


class RNNBinaryClassificator(RNNClassificationRecommender):
    """ Recurrent model for binary classification
        (the goal is to predict if right clickout reference is the first or not in the impressions list)
    """
    
    def __init__(self, dataset, input_shape, cell_type, num_recurrent_layers, num_recurrent_units, num_dense_layers,
                use_generator=False, validation_split=0.15, class_weights=None, sample_weights=None, metrics=['accuracy'],
                optimizer='adam', batch_size=64, checkpoints_path=None, tensorboard_path=None):
        
        super().__init__(dataset=dataset, input_shape=input_shape, cell_type=cell_type, num_recurrent_layers=num_recurrent_layers,
                        num_recurrent_units=num_recurrent_units, num_dense_layers=num_dense_layers, output_size=1,
                        use_generator=use_generator, validation_split=validation_split, metrics=metrics, sample_weights=sample_weights,
                        loss='binary_crossentropy', optimizer=optimizer, class_weights=class_weights, batch_size=batch_size,
                        checkpoints_path=checkpoints_path, tensorboard_path=tensorboard_path)
        
        self.name += '_bin'

    def build_model(self, input_shape, cell_type, num_recurrent_layers, num_recurrent_units, num_dense_layers,
                    bidirectional, output_size, use_batch_normalization):
        CELL = LSTM if cell_type == 'LSTM' else GRU
        
        self.model = Sequential()
        #m.add( TimeDistributed(Dense(64), input_shape=(6,68)) )
        self.model.add( CELL(64, input_shape=input_shape, recurrent_dropout=0.2, dropout=0.2, return_sequences=True) )
        self.model.add( CELL(32, recurrent_dropout=0.2, dropout=0.2, return_sequences=False) )
        self.model.add( Dense(32, activation='relu') )
        self.model.add( Dropout(0.2) )
        self.model.add( Dense(1, activation='sigmoid') )
        self.model.add( Dropout(0.1) )


    def fit(self, epochs, early_stopping_patience=20, early_stopping_on='val_loss', mode='min'):
        super().fit(epochs=epochs, early_stopping_patience=early_stopping_patience, early_stopping_on=early_stopping_on,
                    mode=mode)

    def recommend_batch(self, target_indices):
        pass
        """
        X, indices = self.dataset.load_Xtest()
        
        # predict the references
        predictions = self.model.predict(X)
        
        # take only the last index for each session (target row) and flatten
        indices = indices[:,-1].flatten()
        
        # take only the target predictions
        pred_df = pd.DataFrame(predictions)
        pred_df['orig_index'] = indices
        pred_df = pred_df.set_index('orig_index')
        predictions = pred_df.loc[target_indices]
        del pred_df

        assert len(predictions) == len(target_indices)

        full_df = data.full_df()

        result_predictions = []
        for index in tqdm(target_indices):
            # get the impressions of the clickout to predict
            impr = list(map(int, full_df.loc[index]['impressions'].split('|')))
            # build a list of (impression, score)
            prediction_impressions_distances = [ (impr[j], predictions.at[index,j]) for j in range(len(impr)) ]
            # order the list based on scores (greater is better)
            prediction_impressions_distances.sort(key=lambda tup: tup[1], reverse=True)
            # get only the impressions ids
            ordered_impressions = list(map(lambda x: x[0], prediction_impressions_distances))
            # append the couple (index, reranked impressions)
            result_predictions.append( (index, ordered_impressions) )

        print('prediction created !!!')

        return result_predictions
        """

    def evaluate(self):
        xtest, indices = self.dataset.load_Xtest()

        def get_y_true(clickout_indices):
            df = data.full_df().loc[clickout_indices]
            
            def add_label(row):
                impress = list(map(int, row['impressions'].split('|')))
                ref = row['reference']

                if ref in impress:
                    return 1 if impress[0] == ref else 0
                else:
                    return 0
            
            df = df.astype({'reference':'int'})
            df['label'] = df.progress_apply(add_label, axis=1)
            return df['label']
        
        y_true = get_y_true(indices)
        y_pred = self.model.predict_classes(xtest)

        report = classification_report(y_true, y_pred, target_names=['class1','class0'])
        print(report)
        return report

    def create_feature(self):
        # load dataset and indices
        train, train_indices = self.dataset.load_Xtrain(return_indices=True)
        test, test_indices = self.dataset.load_Xtest()
        # make predictions
        train_test = np.concatenate([train, test])
        del train
        del test
        predictions = self.model.predict(train_test).flatten()
        # build feature df
        concat_indices = np.concatenate([train_indices, test_indices])
        del train_indices
        del test_indices
        users_sessions = data.full_df().loc[concat_indices]
        feature_df = pd.DataFrame({ 'user_id':          users_sessions['user_id'],
                                    'session_id':       users_sessions['session_id'],
                                    'rnn_binary_preds':  predictions },
                                index=concat_indices)
        
        path = 'dataset/preprocessed/no_cluster/{}/feature/rnn_binary_preds/features.csv'.format(self.mode)
        check_folder(path)
        feature_df.to_csv(path)

        return feature_df

    def get_scores_batch(self):
        return None
    
    def get_scores_cv(self, x, groups, test_indices):
        """ Return scores for a fold """
        x_val = x[test_indices]
        indices = x_val[:,:,0][:,self.dataset.rows_per_sample-1]

        predictions = self.model.predict(x_val[:,:,1:])

        # take the target rows
        res_df = data.full_df()[['user_id','session_id']].loc[indices].copy()

        # add the scores as a new column
        res_df['scores'] = predictions

        return res_df


if __name__ == "__main__":
    import utils.menu as menu
    tqdm.pandas()

    def interactive_model(mode, optim='adam'):
        pad = menu.single_choice('Which dataset?', ['Padded 6','Padded 12'], [lambda: 6, lambda: 12])
        dataset = SequenceDatasetForBinaryClassification(f'dataset/preprocessed/cluster_recurrent/{mode}/dataset_binary_classification_p{pad}')

        weights = dataset.get_class_weights()
        
        model = RNNBinaryClassificator(dataset, input_shape=(dataset.rows_per_sample, 168),
                                    cell_type='gru', 
                                    num_recurrent_layers=2, num_recurrent_units=64, num_dense_layers=2,
                                    class_weights=weights, optimizer=optim)

        return model
    
    def train():
        mode = menu.mode_selection()

        # build the model
        opt = menu.single_choice('Optimizer?', ['Adam','RMSProp'], ['adam','rmsprop'])
        lr = menu.single_choice('Learning rate?', ['e-3', 'e-4', 'e-5'], [1e-3, 1e-4, 1e-5])
        if opt == 'adam':
            optim = keras.optimizers.Adam(lr=lr)
        else:
            optim = keras.optimizers.RMSprop(lr=lr)
        
        model = interactive_model(mode, optim=optim)

        # fit the model
        model.fit(epochs=10000)
        print('\nFit completed!')

        best_accuracy = np.max(model.history.history['val_acc'])
        
        model.save(folderpath='saved_models/', suffix='_{}'.format(round(best_accuracy, 5)).replace('.','') )

        # evaluate
        report = model.evaluate()
        bot.send_message(report, account='parro')

        print('Opt: {}'.format(opt))
        print('Lr: {}'.format(lr))

    def create_feature():
        mode = 'full'
        model = interactive_model(mode)

        model_checkpoints = os.listdir('saved_models')
        checkpoint_path = menu.single_choice('Choose the model checkpoint:', model_checkpoints)
        checkpoint_path = os.path.join('saved_models', checkpoint_path)

        print('Loading {}...'.format(checkpoint_path), end='\r', flush=True)
        model.load(checkpoint_path)
        print('Done!',  flush=True)

        print('Creating feature for {}...'.format(mode))
        model.create_feature()

    activity = menu.single_choice('What do you want to do?', ['Train', 'Create feature'], [train, create_feature])