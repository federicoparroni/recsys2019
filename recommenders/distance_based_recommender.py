"""
Base class for a distance based recommender.
Supports several distance metrics, thanks to similaripy library.
See https://github.com/bogliosimone/similaripy/blob/master/guide/temp_guide.md
for documentation and distance formulas
"""
import os
import sys
sys.path.append(os.getcwd())
import time
from recommenders.recommender_base import RecommenderBase
import utils.log as log
import numpy as np
import similaripy as sim
import data
from tqdm import tqdm
import scipy.sparse as sps
import utils.check_folder as cf
import sklearn.preprocessing as prep


class DistanceBasedRecommender(RecommenderBase):
    """
    Base class for a distance based recommender.
    Supports several distance metrics, thanks to similaripy library
    """

    #SIM_DOTPRODUCT = 'dotproduct'
    SIM_COSINE = 'cosine'
    SIM_ASYMCOSINE = 'asymcosine'
    SIM_JACCARD = 'jaccard'
    SIM_DICE = 'dice'
    SIM_TVERSKY = 'tversky'
    SIM_P3ALPHA = 'p3alpha'
    SIM_RP3BETA = 'rp3beta'
    SIM_SPLUS = 'splus'

    def __init__(self, matrix, _type, mode='full', cluster='no_cluster', name='distancebased', urm_name='urm_clickout', k=100, distance='cosine', shrink=0, threshold=0,
                 implicit=False, alpha=0.5, beta=0.5, l=0.5, c=0.5, urm=None, matrix_mul_order='standard'):
        super(DistanceBasedRecommender, self).__init__(mode=mode, cluster=cluster, name=name)
        self.type = _type
        self.urm_name = urm_name
        self._sim_matrix = None
        self.target_indices = data.target_indices(mode, cluster)
        self._matrix_mul_order = matrix_mul_order # if you want R•R', or 'inverse' if you want to compute S•R

        self.dict_row= data.dictionary_row(mode=self.mode, cluster=self.cluster, urm_name=self.urm_name, type=self.type)
        self.dict_col = data.dictionary_col(mode=self.mode, cluster=self.cluster, urm_name=self.urm_name, type=self.type)

        self.matrix = matrix
        self.k = int(k)
        self.distance = distance
        self.shrink = shrink
        self.threshold = threshold
        self.implicit = implicit
        self.alpha = alpha
        self.beta = beta
        self.l = l
        self.c = c
        self.urm = urm
        self.scores_batch = None

    def fit(self):
        self.alpha = -1 if self.alpha is None else self.alpha
        self.beta = -1 if self.beta is None else self.beta
        self.l = -1 if self.l is None else self.l
        self.c = -1 if self.c is None else self.c
        if self.distance==self.SIM_ASYMCOSINE and not(0 <= self.alpha <= 1):
            log.error('Invalid parameter alpha in asymmetric cosine Similarity_MFD!')
            return
        if self.distance==self.SIM_TVERSKY and not(0 <= self.alpha <= 1 and 0 <= self.beta <= 1):
            log.error('Invalid parameter alpha/beta in tversky Similarity_MFD!')
            return
        if self.distance==self.SIM_P3ALPHA and self.alpha is None:
            log.error('Invalid parameter alpha in p3alpha Similarity_MFD')
            return
        if self.distance==self.SIM_RP3BETA and self.alpha is None and self.beta is None:
            log.error('Invalid parameter alpha/beta in rp3beta Similarity_MFD')
            return
        if self.distance==self.SIM_SPLUS and not(0 <= self.l <= 1 and 0 <= self.c <= 1 and 0 <= self.alpha <= 1 and 0 <= self.beta <= 1):
            log.error('Invalid parameter alpha/beta/l/c in s_plus Similarity_MFD')
            return
        
        # compute and stores the Similarity_MFD matrix using one of the distance metric: S = R•R'
        if self.distance==self.SIM_COSINE:
            self._sim_matrix = sim.cosine(self.matrix, k=self.k, shrink=self.shrink, threshold=self.threshold, binary=self.implicit)
        elif self.distance==self.SIM_ASYMCOSINE:
            self._sim_matrix = sim.asymmetric_cosine(self.matrix, k=self.k, shrink=self.shrink, threshold=self.threshold, binary=self.implicit, alpha=self.alpha)
        elif self.distance==self.SIM_JACCARD:
            self._sim_matrix = sim.jaccard(self.matrix, k=self.k, shrink=self.shrink, threshold=self.threshold, binary=self.implicit)
        elif self.distance==self.SIM_DICE:
            self._sim_matrix = sim.dice(self.matrix, k=self.k, shrink=self.shrink, threshold=self.threshold, binary=self.implicit)
        elif self.distance==self.SIM_TVERSKY:
            self._sim_matrix = sim.tversky(self.matrix, k=self.k, shrink=self.shrink, threshold=self.threshold, binary=self.implicit, alpha=self.alpha, beta=self.beta)
        elif self.distance==self.SIM_P3ALPHA:
            self._sim_matrix = sim.p3alpha(self.matrix, k=self.k, shrink=self.shrink, threshold=self.threshold, binary=self.implicit, alpha=self.alpha)
        elif self.distance==self.SIM_RP3BETA:
            self._sim_matrix = sim.rp3beta(self.matrix, k=self.k, shrink=self.shrink, threshold=self.threshold, binary=self.implicit, alpha=self.alpha, beta=self.beta)
        elif self.distance==self.SIM_SPLUS:
            self._sim_matrix = prep.normalize(sim.s_plus(self.matrix, k=self.k, shrink=self.shrink, threshold=self.threshold, binary=self.implicit, l=self.l, t1=self.alpha, t2=self.beta, c=self.c), norm='l2', axis=0)
        else:
            log.error('Invalid distance metric: {}'.format(self.distance))
        return self._sim_matrix
    
    def _has_fit(self):
        """
        Check if the model has been fit correctly before being used
        """
        if self._sim_matrix is None:
            log.error('Cannot recommend without having fit with a proper matrix. Call method \'fit\'.')
            return False
        else:
            return True
    
    def get_r_hat(self):
        """
        Return the r_hat matrix as: R^ = R•S or R^ = S•R
        """
        R = self.urm
        target_indices_urm = []
        for ind in self.target_indices:
            if self.type == 'user':
                target_indices_urm.append(self.dict_row[data.full_df().loc[ind]['user_id']])
            if self.type == 'session':
                target_indices_urm.append(self.dict_row[tuple(data.full_df().loc[ind][['user_id', 'session_id']])])
        if self._matrix_mul_order == 'inverse':
            return self._sim_matrix.tocsr()[target_indices_urm].dot(R)
        else:
            return R[target_indices_urm].dot(self._sim_matrix)

    def get_sim_matrix(self):
        if self._sim_matrix is not None:
            return self._sim_matrix
        else:
            print('NOT TRAINED')

    def recommend_batch(self):
        print('recommending batch')
        if not self._has_fit():
            return None
        full_df = data.full_df()

        # compute the R^ by multiplying: R•S or S•R
        R_hat = self.get_r_hat()

        predictions_batch = []
        scores_batch = []
        count = 0
        predicted_count = 0
        skipped_count = 0
        for index in tqdm(self.target_indices):
            impr = list(map(int, full_df.loc[index]['impressions'].split('|')))

            # get ratings
            l = [[i, R_hat[count, self.dict_col[i]]] for i in impr]
            l.sort(key=lambda tup: tup[1], reverse=True)

            count += 1
            if l[0][1] == 0:
                skipped_count +=1
                continue
            else:
                predicted_count +=1
            recs = [e[0] for e in l]
            scores = [e[1] for e in l]

            predictions_batch.append((index, recs))
            scores_batch.append((index, recs, scores))

        self.scores_batch = scores_batch
        print(f'predicted percentage: {predicted_count / len(self.target_indices)}\n jumped percentage: {skipped_count/len(self.target_indices)}')
        return predictions_batch

    def save_similarity_matrix(self):
        base_save_path = f'dataset/preprocessed/{self.cluster}/{self.mode}/matrices/{self.session}/similarities_matrices'
        cf.check_folder(base_save_path)
        print('saving sim_matrix...')
        sps.save_npz(f'{base_save_path}/{self.urm_name}_{self.name}', self.get_sim_matrix())
        print('sim_matrix saved succesfully !')

    def save_r_hat(self):
        base_save_path = f'dataset/preprocessed/{self.cluster}/{self.mode}/matrices/{self.session}/r_hat_matrices'
        cf.check_folder(base_save_path)
        print('saving r_hat...')
        sps.save_npz(f'{base_save_path}/{self.urm_name}_{self.name}', self.get_r_hat())
        print('r_hat saved succesfully !')

    def get_scores_batch(self):
        if self.scores_batch is None:
            self.recommend_batch()

        return self.scores_batch




