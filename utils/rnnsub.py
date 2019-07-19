import os
import sys
sys.path.append(os.getcwd())

import data
import out
import utils.menu as menu
from recommenders.recurrent.RNNClassificationRecommender import RNNClassificationRecommender
from utils.dataset import SequenceDatasetForClassification


if __name__ == "__main__":
    mode = menu.mode_selection()
    
    dataset = SequenceDatasetForClassification(f'dataset/preprocessed/cluster_recurrent/{mode}/dataset_classification')

    model = RNNClassificationRecommender(dataset, cell_type='gru', num_recurrent_units=128, num_recurrent_layers=3,
                                     num_dense_layers=2, class_weights=[])

    model.load('/Users/federico/Desktop/rnn_GRU_3layers_128units_2dense_class_06316.h5')
    #model.load('gru.h5')

    target_indices = data.target_indices(mode, 'cluster_recurrent')
    recomendations = model.recommend_batch(target_indices)

    if mode != 'full':
        model.compute_MRR(recomendations)
    
    out.create_sub(recomendations, submission_name=model.name + '_06316')

