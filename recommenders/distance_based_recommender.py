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

from functools import partial
import pathos.pools as pp
import multiprocessing


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

    def __init__(self, mode='full', urm_name='urm_clickout'):
        super(DistanceBasedRecommender, self).__init__(mode=mode, urm_name=urm_name)
        self.name = 'distancebased'
        self.mode = mode
        self.urm_name = urm_name
        self._sim_matrix = None
        self._matrix_mul_order = 'standard' # if you want R•R', or 'inverse' if you want to compute S•R
        self.R_hat = None
        self.dict_col = None

    def fit(self, matrix, k, distance, shrink=0, threshold=0, implicit=True, alpha=0.5, beta=0.5, l=0.5, c=0.5, urm=None):
        """
        Initialize the model and compute the Similarity_MFD matrix S with a distance metric.
        Access the Similarity_MFD matrix using: self._sim_matrix

        Parameters
        ----------
        matrix : csr_matrix
            A sparse matrix. For example, it can be the URM of shape (number_users, number_items).
        k : int
            K nearest neighbour to consider.
        distance : str
            One of the supported distance metrics, check collaborative_filtering_base constants.
        shrink : float, optional
            Shrink term used in the normalization
        threshold: float, optional
            All the values under this value are cutted from the final result
        implicit: bool, optional
            If true, treat the URM as implicit, otherwise consider explicit ratings (real values) in the URM
        alpha: float, optional, included in [0,1]
        beta: float, optional, included in [0,1]
        l: float, optional, balance coefficient used in s_plus distance, included in [0,1]
        c: float, optional, cosine coefficient, included in [0,1]
        urm: urm that has to be multiplied by the similarity matrix
        """
        self.urm = urm.tocsc() # debug
        alpha = -1 if alpha is None else alpha
        beta = -1 if beta is None else beta
        l = -1 if l is None else l
        c = -1 if c is None else c
        if distance==self.SIM_ASYMCOSINE and not(0 <= alpha <= 1):
            log.error('Invalid parameter alpha in asymmetric cosine Similarity_MFD!')
            return
        if distance==self.SIM_TVERSKY and not(0 <= alpha <= 1 and 0 <= beta <= 1):
            log.error('Invalid parameter alpha/beta in tversky Similarity_MFD!')
            return
        if distance==self.SIM_P3ALPHA and alpha is None:
            log.error('Invalid parameter alpha in p3alpha Similarity_MFD')
            return
        if distance==self.SIM_RP3BETA and alpha is None and beta is None:
            log.error('Invalid parameter alpha/beta in rp3beta Similarity_MFD')
            return
        if distance==self.SIM_SPLUS and not(0 <= l <= 1 and 0 <= c <= 1 and 0 <= alpha <= 1 and 0 <= beta <= 1):
            log.error('Invalid parameter alpha/beta/l/c in s_plus Similarity_MFD')
            return
        
        # compute and stores the Similarity_MFD matrix using one of the distance metric: S = R•R'
        if distance==self.SIM_COSINE:
            self._sim_matrix = sim.cosine(matrix, k=k, shrink=shrink, threshold=threshold, binary=implicit)
        elif distance==self.SIM_ASYMCOSINE:
            self._sim_matrix = sim.asymmetric_cosine(matrix, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha)
        elif distance==self.SIM_JACCARD:
            self._sim_matrix = sim.jaccard(matrix, k=k, shrink=shrink, threshold=threshold, binary=implicit)
        elif distance==self.SIM_DICE:
            self._sim_matrix = sim.dice(matrix, k=k, shrink=shrink, threshold=threshold, binary=implicit)
        elif distance==self.SIM_TVERSKY:
            self._sim_matrix = sim.tversky(matrix, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha, beta=beta)
        elif distance==self.SIM_P3ALPHA:
            self._sim_matrix = sim.p3alpha(matrix, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha)
        elif distance==self.SIM_RP3BETA:
            self._sim_matrix = sim.rp3beta(matrix, k=k, shrink=shrink, threshold=threshold, binary=implicit, alpha=alpha, beta=beta)
        elif distance==self.SIM_SPLUS:
            self._sim_matrix = sim.s_plus(matrix, k=k, shrink=shrink, threshold=threshold, binary=implicit, l=l, t1=alpha, t2=beta, c=c)
        else:
            log.error('Invalid distance metric: {}'.format(distance))
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
    
    def get_r_hat(self, verbose=False):
        """
        Return the r_hat matrix as: R^ = R•S or R^ = S•R
        """
        R = self.urm
        targetids = data.target_urm_rows(self.mode)
        if self._matrix_mul_order == 'inverse':
            return self._sim_matrix.tocsr()[targetids].dot(R)
        else:
            return R[targetids].dot(self._sim_matrix)

    def get_sim_matrix(self):
        if self._sim_matrix is not None:
            return self._sim_matrix
        else:
            print('NOT TRAINED')

    def recommend_batch(self, verbose=False):
       	print('recommending batch')
        if not self._has_fit():
            return None
        df_handle = data.handle_df(mode=self.mode)
        #dict_row = data.dictionary_row(mode=self.mode)
        dict_col = data.dictionary_col(mode=self.mode)

        # compute the R^ by multiplying: R•S or S•R
        self.R_hat = self.get_r_hat(verbose)
        print("R_hat computed")
        
        #target_rows = data.target_urm_rows(self.mode)
        predictions = []
        for index, row in tqdm(df_handle.iterrows()):
            impr = list(map(int, row['impressions'].split('|')))
            l = [[i, self.R_hat[index, dict_col[i]]] for i in impr]
            l.sort(key=lambda tup: tup[1], reverse=True)
            predictions.append((row['session_id'], [e[0] for e in l]))

        return predictions


    def multi_thread_recommend_batch(self, verbose=False):
        print('recommending batch')

        if not self._has_fit():
            return None

        df_handle = data.handle_df(mode=self.mode)
        self.dict_col = data.dictionary_col(mode=self.mode)

        # compute the R^ by multiplying: R•S or S•R
        self.R_hat = self.get_r_hat(verbose)
        print("R_hat computed")

        predictions = []

        """
        multiprocessing part
        """
        # instance a number of workers equal to the cpu of the machine
        workers = multiprocessing.cpu_count()

        #define the call to the function _recommend_row passing the fix parameter (in this case R_hat)

        #create a pool with a number of processes equal to the cpu count
        pool = pp.ProcessPool(workers)

        # add the indexes as the first column of the dataframe
        df_handle.reset_index(inplace=True)

        # convert the handle from dataframe to numpy array
        handle_array = df_handle.values

        start = time.time()
        # start the pool passing to the function with the fixed input the remaining variable parameter (handle_array)
        predictions = pool.map(self._recommend_row, handle_array)
        pool.close()
        pool.join()
        print("recommendations created in {}:".format(time.time()-start))


        return predictions


    def _recommend_row(self, row):

        #TODO: HAVE TO BE CHANGED
        print(row[0])
        impr = list(map(int, row[6].split('|')))
        l = [[i, self.R_hat[row[0], self.dict_col[i]]] for i in impr]
        l.sort(key=lambda tup: tup[1], reverse=True)
        return row['session_id'], [e[0] for e in l]

    def run(self):
        pass
