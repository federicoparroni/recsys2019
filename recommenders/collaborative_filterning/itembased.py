"""
Collaborative filtering recommender.
"""
from recommenders.distance_based_recommender import DistanceBasedRecommender
import similaripy as sim
import numpy as np
from bayes_opt import BayesianOptimization
import time
import utils.dated_directory as datedir
import scipy.sparse as sps
import out
import data

class CFItemBased(DistanceBasedRecommender):
    """
    Computes the recommendations for a user by looking for the similar users based on the
    item which they rated
    """

    def __init__(self, mode='full', urm_name='urm_clickout', k=100, distance='cosine', shrink=0, 
                 threshold=0, implicit=True, alpha=0.5, beta=0.5, l=0.5, c=0.5):
        """
        Initialize the model

        Parameters
        ----------
        k: int
            K nearest neighbour to consider.
        distance: str
            One of the supported distance metrics, check collaborative_filtering_base constants.
        shrink: float, optional
            Shrink term used in the normalization
        threshold: float, optional
            All the values under this value are cutted from the final result
        implicit: bool, optional
            If true, treat the URM as implicit, otherwise consider explicit ratings (real values) in the URM
        alpha: float, optional, included in [0,1]
        beta: float, optional, included in [0,1]
        l: float, optional, balance coefficient used in s_plus distance, included in [0,1]
        c: float, optional, cosine coefficient, included in [0,1]
        """
        urm = data.urm(mode, urm_name=urm_name)
        self.urm = data.urm(mode, urm_name=urm_name)
        super(CFItemBased, self).__init__(self.urm.T,
                                          mode=mode, 
                                          urm_name=urm_name, 
                                          k=k, 
                                          distance=distance, 
                                          shrink=shrink, 
                                          threshold=threshold, 
                                          implicit=implicit, 
                                          alpha=alpha, 
                                          beta=beta,
                                          l=l,
                                          c=c,
                                          urm=self.urm)
        
        self.name = 'CFitem: k: {} distance: {} shrink: {} threshold: {} implicit: {} alpha: {} beta: {} l: {} c: {}'.format(k,
                                                                                                                             distance,
                                                                                                                             shrink,
                                                                                                                             threshold,
                                                                                                                             implicit,
                                                                                                                             alpha,
                                                                                                                             beta,
                                                                                                                             l,
                                                                                                                             c)

