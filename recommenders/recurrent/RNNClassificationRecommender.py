import sys
import os
sys.path.append(os.getcwd())

import data
import pandas as pd
import numpy as np

from recommenders.RNN import RecurrentRecommender

from utils.dataset import SequenceDatasetForClassification

from utils.check_folder import check_folder
from tqdm import tqdm


class RNNClassificationRecommender(RecurrentRecommender):
    """ Recurrent model for classification
        (the goal is to predict the index of the right clickout reference in the impressions list)
    """
    
    def __init__(self, dataset, cell_type, num_recurrent_layers, num_recurrent_units, num_dense_layers,
                use_generator=True, validation_split=0.15, class_weights=None,
                optimizer='rmsprop', checkpoints_path=None, tensorboard_path=None):
        
        super().__init__(dataset=dataset, cell_type=cell_type, num_recurrent_layers=num_recurrent_layers,
                        num_recurrent_units=num_recurrent_units, num_dense_layers=num_dense_layers, output_size=25,
                        use_generator=use_generator, validation_split=validation_split,
                        loss='categorical_crossentropy', optimizer=optimizer, class_weights=class_weights,
                        checkpoints_path=checkpoints_path, tensorboard_path=tensorboard_path)
        
        self.name += '_class'
        

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

    def get_scores_batch(self):
        return None


if __name__ == "__main__":
    import utils.menu as menu

    # build the weights array
    weights = np.array([0.37738,0.10207,0.07179,0.05545,0.04711,0.03822,0.03215,0.02825,0.02574,
                        0.02289,0.02239,0.02041,0.01814,0.01619,0.01451,0.01306,0.01271,0.01156,
                        0.01174,0.01072,0.01018,0.00979,0.00858,0.00868,0.01029])
    weights = 1/weights
    wgt_sum = sum(weights)
    weights = weights/wgt_sum

    mode = menu.mode_selection()
    cell_type = menu.single_choice('Choose a network architecture:', ['LSTM', 'GRU', 'default architecture'], [lambda: 'LSTM', lambda: 'GRU', lambda: 'auto'])
    print()
    if cell_type == 'auto':
        cell_type = 'GRU'
        epochs = 1
        rec_layers = 1
        dense_layers = 2
        units = 4
    else:
        epochs = int(input('Insert number of epochs: '))
        rec_layers = int(input('Insert number of recurrent layers: '))
        units = int(input('Insert number of units per layer: '))
        dense_layers = int(input('Insert number of dense layers: '))
        weights = menu.yesno_choice('Do you want to use samples weighting?', lambda: weights, lambda: [])
        #tb_path = menu.yesno_choice('Do you want to enable Tensorboard?', lambda: 'recommenders/tensorboard', lambda: None)
    tb_path = None

    dataset = SequenceDatasetForClassification(f'dataset/preprocessed/cluster_recurrent/{mode}/dataset_classification')
    
    model = RNNClassificationRecommender(dataset, use_generator=True, cell_type=cell_type,
                                        num_recurrent_layers=rec_layers, num_recurrent_units=units,
                                        num_dense_layers=dense_layers, class_weights=weights)
    model.fit(epochs=epochs)

    print('\nFit completed!')

    target_indices = data.target_indices(mode, 'cluster_recurrent')
    
    recommendations = model.recommend_batch(target_indices)
    #print(recommendations)
    print('Recommendation count: ', len(recommendations))
    
    model.compute_MRR(recommendations)

    #menu.yesno_choice('Do you want to save the model?', lambda: model.save(folderpath='.'))
    model.save(folderpath='.')
