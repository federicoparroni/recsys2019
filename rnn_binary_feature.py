import utils.menu as menu
from utils.dataset import SequenceDatasetForBinaryClassification
from recommenders.recurrent.RNNBinaryClassificator import RNNBinaryClassificator

if __name__ == "__main__":
    
    mode = menu.mode_selection()
    
    dataset = SequenceDatasetForBinaryClassification(f'dataset/preprocessed/cluster_recurrent/{mode}/dataset_binary_classification_p12')

    model = RNNBinaryClassificator(dataset, input_shape=(dataset.rows_per_sample, 118), cell_type='gru', 
                                num_recurrent_units=64, num_recurrent_layers=2,
                                num_dense_layers=1, class_weights=[])
    
    file_path = input('Insert model path: ')
    model.load(file_path)

    model.create_feature()