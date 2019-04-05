"""
Collaborative filtering recommender.
"""
from recommenders.distance_based_recommender import DistanceBasedRecommender
import data
import sklearn.preprocessing as preprocessing
#from validator import BayesianValidator

class CFItemBased(DistanceBasedRecommender):
    """
    Computes the recommendations for a user by looking for the similar users based on the
    item which they rated
    """

    def __init__(self, mode='full', cluster='no_cluster', urm_name='urm_clickout', k=100, distance='splus', shrink=0,
                 threshold=0, implicit=False, alpha=0.5, beta=0.5, l=0.5, c=0.5):
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

        urm = data.urm(mode, cluster, urm_name=urm_name)

        # create fixed params dictionary
        self.fixed_params_dict = {
            'mode': mode,
            'urm_name': urm_name,
            'distance': distance,
            'implicit': implicit,
            'threshold': 0,
            'cluster': cluster
        }

        # create hyperparameters dictionary
        self.hyperparameters_dict = {
            'k': (2000, 10000),
            'beta': (0, 1),
            'alpha': (0, 1),
            'l': (0, 1),
            'shrink': (0, 5),
            'c': (0, 1)
        }

        super(CFItemBased, self).__init__(matrix=urm.T,
                                          mode=mode,
                                          cluster= cluster,
                                          urm_name=urm_name,
                                          name='ItemKNN: urm: {} k: {} distance: {} shrink: {} threshold: {} implicit: {}'
                                               ' alpha: {} beta: {} l: {} c: {}'.format(urm_name, k, distance, shrink, threshold,
                                                                                        implicit, alpha, beta, l, c),
                                          k=k, 
                                          distance=distance, 
                                          shrink=shrink, 
                                          threshold=threshold, 
                                          implicit=implicit, 
                                          alpha=alpha, 
                                          beta=beta,
                                          l=l,
                                          c=c,
                                          urm=urm,
                                          matrix_mul_order='standard')




if __name__ == '__main__':
    recommender = CFItemBased(mode='small', urm_name='urm_session_aware_lin')
    val = BayesianValidator(recommender)
    val.validate(iterations=100)
