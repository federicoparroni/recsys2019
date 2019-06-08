from recommenders.recommender_base import RecommenderBase
import os
import numpy as np
from tqdm import tqdm
import data
import utils.check_folder as cf
from preprocess_utils.last_clickout_indices import find as find_last_clickouts


class TensorflowRankig(RecommenderBase):

    def __init__(self, mode, cluster, dataset_name, pred_name, predict_train = False):
        """
        the dataset name is used to load the prediction created by the tensorflow ranking class

        :param dataset_name: dataset name passed to the CreateDataset() method
        """

        name = 'tf_ranking'
        super(TensorflowRankig, self).__init__(mode=mode, cluster=cluster, name=name)

        self.dataset_name = dataset_name

        # the path where the PREDICTION are stored
        _BASE_PATH = f'dataset/preprocessed/tf_ranking/{cluster}/{mode}/{self.dataset_name}'

        _PREDICTION_PATH = f'{_BASE_PATH}/{pred_name}.npy'

        # check if the PREDICTION have been made
        exists_path_predictions = os.path.isfile(_PREDICTION_PATH)

        if not exists_path_predictions:
            print(f'the prediction for the \ndataset: {self.dataset_name}\n mode:{mode}\n '
                  f'cluster:{cluster}\n have not been made')
            exit(0)

        self.predictions = np.load(_PREDICTION_PATH)
        print('predictions loaded')
        if not predict_train:
            self.target_indices = data.target_indices(mode, cluster)
        else:
            self.target_indices = sorted(find_last_clickouts(data.full_df()))

    def fit(self):
        pass

    def recommend_batch(self):

        final_predictions = []
        scores_batch=[]

        count = 0
        for index in tqdm(self.target_indices):
            impr = list(map(int, data.full_df().loc[index]['impressions'].split('|')))
            pred = self.predictions[count][0:len(impr)]
            couples = list(zip(pred, impr))
            #print(couples)
            couples.sort(key=lambda x: x[0], reverse=True)
            scores, sorted_impr = zip(*couples)
            final_predictions.append((index, list(sorted_impr)))
            scores_batch.append((index, list(sorted_impr), list(scores)))
            count += 1
        if self.mode != 'small':
            cf.check_folder('scores')
            np.save(f'scores/{self.name}', np.array(scores_batch))
        return final_predictions

    def get_scores_batch(self):
        final_predictions = []

        count = 0
        for index in tqdm(self.target_indices):
            impr = list(map(int, data.full_df().loc[index]['impressions'].split('|')))
            pred = self.predictions[count][0:len(impr)]
            couples = list(zip(pred, impr))
            # print(couples)
            couples.sort(key=lambda x: x[0], reverse=True)
            scores, sorted_impr = zip(*couples)
            final_predictions.append((index, list(sorted_impr), list(scores)))
            count += 1
        return final_predictions




if __name__ == '__main__':
    recommender = TensorflowRankig(mode='small', cluster='no_cluster', dataset_name='prova3')
    recommender.evaluate(send_MRR_on_telegram=True)


