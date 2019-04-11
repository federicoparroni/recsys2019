from recommenders.recommender_base import RecommenderBase
import data
from tqdm import tqdm
import numpy as np
import sklearn.preprocessing as sps
import similaripy as sim


class ContentBased(RecommenderBase):

    def __init__(self, mode, urm_name, _type, cluster='no_cluster'):
        name = 'content_based_remastered'

        super(ContentBased, self).__init__(mode, cluster, name)
        self.type = _type
        self.urm_name = urm_name
        self.dict_row = data.dictionary_row(mode=mode, urm_name=urm_name, cluster=cluster, type=self.type)
        self.dict_col = data.dictionary_col(mode=mode, urm_name=urm_name, cluster=cluster, type=self.type)
        self.user_features_matrix = None
        self.scores_batch = None

        # load target indices
        self.target_indices = data.target_indices(mode, cluster)

    def fit(self):

        urm = data.urm(self.mode, self.cluster, self.type, self.urm_name)
        icm = data.icm().tocsr()

        # computing target indices_urm
        target_indices_urm = []
        if self.type == 'user':
            for ind in self.target_indices:
                target_indices_urm.append(self.dict_row[data.full_df().loc[ind]['user_id']])
        if self.type == 'session':
            for ind in self.target_indices:
                target_indices_urm.append(self.dict_row[tuple(data.full_df().loc[ind][['user_id', 'session_id']])])

        self.user_features_matrix = sps.normalize(urm[target_indices_urm] * icm, norm='l2', axis=0)

    def recommend_batch(self):
        # load full df
        print('loading df_full')
        full_df = data.full_df()
        icm = data.icm().tocsr() #sim.normalization.bm25(data.icm().tocsr(), axis=1)

        predictions_batch = []
        self.scores_batch = []

        count = 0
        predicted_count = 0
        skipped_count = 0
        for index in tqdm(self.target_indices):

            # get the impressions of the clickout to predict
            impr = list(map(int, full_df.loc[index]['impressions'].split('|')))
            # get the row index of the icm to predict
            icm_rows = []
            for i in impr:
                icm_rows.append(self.dict_col[i])
            temp = list(zip(impr, icm_rows))
            temp.sort(key=lambda tup: tup[1])
            list_impr_icmrows = list(zip(*temp))
            impr_sorted = list(list_impr_icmrows[0])
            icm_rows = list(list_impr_icmrows[1])

            icm_filtered = icm[icm_rows]
            r_hat_row = self.user_features_matrix[count]*icm_filtered.T

            l = list(zip(impr_sorted, r_hat_row.todense().tolist()[0]))
            l_scores = l.copy()
            l.sort(key=lambda tup: tup[1], reverse=True)
            l_scores.sort(key=lambda tup: tup[0], reverse=True)

            count += 1

            if l[0][1] == 0:
                skipped_count += 1
                self.scores_batch.append((index, [], []))
                continue
            else:
                predicted_count += 1
                p = [e[0] for e in l]
                print(f'impr: {impr}\n rec: {p}')
                predictions_batch.append((index, p))

                scores = [e[1] for e in l]
                self.scores_batch.append((index, p, scores))

                print(scores)

        print(f'predicted percentage: {predicted_count/len(self.target_indices)}\n jumped percentage: {skipped_count/len(self.target_indices)}')
        print('prediction created !!!')
        return predictions_batch

    def get_scores_batch(self):
        if self.scores_batch is None:
            self.recommend_batch()
        return self.scores_batch
