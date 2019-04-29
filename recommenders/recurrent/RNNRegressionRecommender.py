import sys
import os
sys.path.append(os.getcwd())

import data
import pandas as pd
import numpy as np

from recommenders.RNN import RecurrentRecommender

from utils.dataset import SequenceDatasetForRegression
from numpy.linalg import norm as L2Norm

from utils.check_folder import check_folder
from tqdm import tqdm


class RNNRegressionRecommender(RecurrentRecommender):
    """ Recurrent model for regression
        (the goal is to predict the features of the right clickout reference)
    """

    def __init__(self, dataset, cell_type, num_units, num_layers, use_generator=True, validation_split=0.15,
                 optimizer='rmsprop', class_weights=None, checkpoints_path=None, tensorboard_path=None):
        
        super().__init__(dataset=dataset, cell_type=cell_type, num_units=num_units, num_layers=num_layers,
                        use_generator=use_generator, validation_split=validation_split, loss='mean_squared_error',
                        optimizer=optimizer, checkpoints_path=checkpoints_path, tensorboard_path=tensorboard_path)

        self.name += '_regr'


    def recommend_batch(self, target_indices):
        X, indices = self.dataset.load_Xtest()
        
        # predict the references
        predictions = self.model.predict(X)
        
        # flatten the predictions and the indices to be 2-dimensional
        predictions = predictions.reshape((-1, predictions.shape[-1]))
        indices = indices.flatten()
        
        # take only the target predictions
        pred_df = pd.DataFrame(predictions)
        pred_df['orig_index'] = indices
        pred_df = pred_df.set_index('orig_index')
        predictions = pred_df.loc[target_indices].sort_index().values
        del pred_df

        assert len(predictions) == len(target_indices)

        train_df = data.train_df('full')
        accomodations1hot_df = data.accomodations_one_hot()

        result_predictions = []
        for k,index in tqdm(enumerate(target_indices)):
            # get the impressions of the clickout to predict
            impr = list(map(int, train_df.loc[index]['impressions'].split('|')))
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

    def get_scores_batch(self):
        return None


if __name__ == "__main__":
    import utils.menu as menu

    mode = menu.mode_selection()
    cell_type = menu.single_choice('Choose a network architecture:', ['LSTM', 'GRU', 'default architecture'], [lambda: 'LSTM', lambda: 'GRU', lambda: 'auto'])
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

    dataset = SequenceDatasetForRegression(f'dataset/preprocessed/cluster_recurrent/{mode}/dataset_regression')
    
    model = RNNRegressionRecommender(dataset, use_generator=True, cell_type=cell_type, num_layers=layers, num_units=units)
    model.fit(epochs=epochs)

    print('\nFit completed!')

    target_indices = data.target_indices(mode, 'cluster_recurrent')
    
    recommendations = model.recommend_batch(target_indices)
    #print(recommendations)
    print('Recommendation count: ', len(recommendations))
    
    model.compute_MRR(recommendations)

    model.save(folderpath='.')