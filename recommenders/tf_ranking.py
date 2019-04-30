from recommenders.recommender_base import RecommenderBase
import os
import numpy as np
from tqdm import tqdm
import data


class TensorflowRankig(RecommenderBase):

    def __init__(self, mode, cluster, dataset_name):
        """
        the dataset name is used to load the prediction created by the tensorflow ranking class

        :param dataset_name: dataset name passed to the CreateDataset() method
        """

        name = 'tf_ranking'
        super(TensorflowRankig, self).__init__(mode=mode, cluster=cluster, name=name)

        self.dataset_name = dataset_name

        # the path where the PREDICTION are stored
        _BASE_PATH = f'dataset/preprocessed/tf_ranking/{cluster}/{mode}/{self.dataset_name}'

        _PREDICTION_PATH = f'{_BASE_PATH}/predictions.npy'
        _TARGET_INDICES_PATH = f'{_BASE_PATH}/target_indices.npy'

        # check if the PREDICTION have been made
        exists_path_predictions = os.path.isfile(_PREDICTION_PATH)
        exists_path_indices = os.path.isfile(_TARGET_INDICES_PATH)

        if (not exists_path_indices) or (not exists_path_predictions):
            print(f'the prediction or the target indices for the \ndataset: {self.dataset_name}\n mode:{mode}\n '
                  f'cluster:{cluster}\n have not been made')
            exit(0)

        self.predictions = np.load(_PREDICTION_PATH)
        print('predictions loaded')

        self.target_indices = np.load(_TARGET_INDICES_PATH)
        print('target indices loaded')

    def fit(self):
        pass

    def recommend_batch(self):

        final_predictions = []

        count = 0
        for index in tqdm(self.target_indices):
            impr = list(map(int, data.full_df().loc[index]['impressions'].split('|')))
            pred = self.predictions[count][0:len(impr)]
            couples = list(zip(pred, impr))
            print(couples)
            couples.sort(key=lambda x: x[0], reverse=True)
            _, sorted_impr = zip(*couples)
            final_predictions.append((index, list(sorted_impr)))
            count += 1
        return final_predictions

    def get_scores_batch(self):
        #TODO: IMPLEMENT GET_SCORES_BATCH
        pass


if __name__ == '__main__':
    recommender = TensorflowRankig(mode='small', cluster='no_cluster', dataset_name='prova')
    recommender.evaluate(send_MRR_on_telegram=True)


