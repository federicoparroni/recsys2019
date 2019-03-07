"""
Collaborative filtering recommender.
"""
from recommenders.distance_based_recommender import DistanceBasedRecommender
import similaripy as sim
import numpy as np
from bayes_opt import BayesianOptimization
# from inout.importexport import exportcsv
import time
import utils.dated_directory as datedir
import scipy.sparse as sps
import out

class CFItemBased(DistanceBasedRecommender):
    """
    Computes the recommendations for a user by looking for the similar users based on the
    item which they rated
    """

    def __init__(self):
        super(CFItemBased, self).__init__()
        self.name = 'CFitem'

    def fit(self, urm_train, k, distance, shrink=0, threshold=0, implicit=True, alpha=None, beta=None, l=None, c=None, verbose=False, urm=None):
        """
        Initialize the model and compute the Similarity_MFD matrix S with a distance metric.
        Access the Similarity_MFD matrix using: self._sim_matrix

        Parameters
        ----------
        urm_train: csr_matrix
            The URM matrix of shape (number_users, number_items) to train the model with.
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
        self.urm = urm_train
        return super(CFItemBased, self).fit(urm_train.T, k=k, distance=distance, shrink=shrink, threshold=threshold,
                                            implicit=implicit, alpha=alpha, beta=beta, l=l, c=c, urm=urm)

    def get_r_hat(self, verbose=False):
        """
        Return the r_hat matrix as: R^ = R•S, ONLY for the TARGET USERS
        """
        return super(CFItemBased, self).get_r_hat(verbose=verbose)
    
    def run(self, distance, urm_train=None, urm_test=None, targetids=None, k=100, shrink=10, threshold=0,
            implicit=True, alpha=None, beta=None, l=None, c=None, with_scores=False, export=True, verbose=True):
        """
        Run the model and export the results to a file

        Parameters
        ----------
        distance : str, distance metric
        urm_train : csr matrix, URM. If None, used: data.get_urm_train(). This should be the
            entire URM for which the targetids corresponds to the row indexes.
        urm_test : csr matrix, urm where to test the model. If None, use: data.get_urm_test()
        targetids : list, target user ids. If None, use: data.get_target_playlists()
        k : int, K nearest neighbour to consider
        shrink : float, shrink term used in the normalization
        threshold : float, all the values under this value are cutted from the final result
        implicit : bool, if true, treat the URM as implicit, otherwise consider explicit ratings (real values) in the URM

        Returns
        -------
        recs: (list) recommendations
        map10: (float) MAP10 for the provided recommendations
        """
        start = time.time()

        urm_train = data.get_urm_train_1() if urm_train is None else urm_train
        urm_test = data.get_urm_test_1() if urm_test is None else urm_test
        targetids = data.get_target_playlists() if targetids is None else targetids

        self.fit(urm_train, k=k, distance=distance, alpha=alpha, beta=beta, c=c, l=l, shrink=shrink, threshold=threshold, implicit=implicit)
        recs = self.recommend_batch(targetids, urm=urm_train, with_scores=with_scores, verbose=verbose)

        map10 = None
        if len(recs) > 0:
            map10 = self.evaluate(recs, test_urm=urm_test, verbose=verbose)
        else:
            log.warning('No recommendations available, skip evaluation')

        if export:
            exportcsv(recs, path='submission', name='{}_{}'.format(self.name,distance), verbose=verbose)

        if verbose:
            log.info('Run in: {:.2f}s'.format(time.time()-start))
        
        return recs, map10

    #def test(self, distance=DistanceBasedRecommender.SIM_SPLUS, k=200, shrink=0, threshold=0, implicit=True, alpha=0.5, beta=0.5, l=0.5, c=0.5):
    def test(self, distance=DistanceBasedRecommender.SIM_SPLUS, k=600, shrink=10, threshold=0, implicit=True, alpha=0.25, beta=0.5, l=0.25, c=0.5):
        """
        Test the model without saving the results. Default distance: SPLUS
        """
        return self.run(distance=distance, k=k, shrink=shrink, threshold=threshold, implicit=implicit, alpha=alpha, beta=beta, l=l, c=c, export=False)

    def validateStep(self, k, shrink, alpha, beta, l, c, threshold):
        # gather saved parameters from self
        distance = self._validation_dict['distance']
        targetids = self._validation_dict['targetids']
        urm_train = self._validation_dict['urm_train']
        urm_test = self._validation_dict['urm_test']
        N = self._validation_dict['N']
        filter_already_liked = self._validation_dict['filter_already_liked']
        items_to_exclude = self._validation_dict['items_to_exclude']
        with_scores = self._validation_dict['with_scores']
        implicit = self._validation_dict['implicit']
        verbose = self._validation_dict['verbose']

        self.fit(urm_train=urm_train, k=int(k), distance=distance, shrink=int(shrink), threshold=int(threshold),
                alpha=alpha, beta=beta, l=l, c=c, implicit=implicit, verbose=verbose)
        # evaluate the model with the current weigths
        recs = self.recommend_batch(userids=targetids, N=N, urm=urm_train, filter_already_liked=filter_already_liked,
                                    with_scores=with_scores, items_to_exclude=items_to_exclude, verbose=verbose)
        return self.evaluate(recs, test_urm=urm_test)

    def validate(self, iterations, urm_train, urm_test, distance, targetids,
                k, shrink, alpha, beta, l, c, threshold=(0,0), N=10, implicit=True,
                filter_already_liked=True, with_scores=False, items_to_exclude=[], verbose=False):
        """
        Validate the models, using the specified intervals for the parameters. Example:
            k: (10,800),
            shrink: (0,100),
            alpha: (0,1),
            beta: (0,1),
            l: (0,1),
            c: (0,1),
            threshold=(0,5)
        """
        # save the params in self to collect them later
        self._validation_dict = {
            'distance': distance,
            'targetids': targetids,
            'urm_train': urm_train,
            'urm_test': urm_test,
            'N': N,
            'with_scores': with_scores,
            'filter_already_liked': filter_already_liked,
            'items_to_exclude': items_to_exclude,
            'implicit': implicit,
            'verbose': verbose
        }

        pbounds = {
            'k': k if isinstance(k, tuple) else (int(k),int(k)),
            'shrink': shrink if isinstance(shrink, tuple) else (int(shrink),int(shrink)),
            'alpha': alpha if isinstance(alpha, tuple) else (float(alpha),float(alpha)),
            'beta': beta if isinstance(beta, tuple) else (float(beta),float(beta)),
            'l': l if isinstance(l, tuple) else (float(l),float(l)),
            'c': c if isinstance(c, tuple) else (float(c),float(c)),
            'threshold': threshold if isinstance(threshold, tuple) else (int(threshold),int(threshold))
        }

        optimizer = BayesianOptimization(
            f=self.validateStep,
            pbounds=pbounds,
            random_state=1
        )
        optimizer.maximize(
            init_points=2,
            n_iter=iterations
        )

        log.warning('Max found: {}'.format(optimizer.max))
        return optimizer

"""
If this file is executed, test the SPLUS distance metric
"""
if __name__ == '__main__':
    import pandas as pd
    import data
    import scipy.sparse as sps
    from preprocess.create_matrices import urm
    handle_df = data.handle_df()
    urm_train = data.train_urm()
    dictionary_row = data.dictionary_row()
    dictionary_col = data.dictionary_col()
    ib = CFItemBased()
    ib.fit(urm_train, 50, ib.SIM_JACCARD, shrink=10, threshold=0, implicit=False, alpha=0.5, beta=0.5, l=0.5, c=0.5, urm=urm_train)
    predictions = ib.recommend_batch(handle_df, dictionary_row, dictionary_col)
    out.create_sub(predictions, handle_df)

    # print()
    # log.success('++ What do you want to do? ++')
    # log.warning('(t) Test the model with some default params')
    # log.warning('(r) Save the R^')
    # log.warning('(s) Save the similarity matrix')
    # log.warning('(v) Validate the model')
    # log.warning('(x) Exit')
    # arg = input()[0]
    # print()
    
    # model = CFItemBased()
    # if arg == 't':
    #     # recs = model.recommend_batch(userids=data.get_target_playlists(), urm=data.get_urm_train())
    #     # model.evaluate(recommendations=recs, test_urm=data.get_urm_test())
    #     model.test(distance=CFItemBased.SIM_P3ALPHA, k=500,alpha=1.7,beta=0.5,shrink=0,l=0.25,c=0.5)
    # elif arg == 'r':
    #     log.info('Wanna save for evaluation (y/n)?')
    #     choice = input()[0] == 'y'
    #     model.fit(data.get_urm_train_2(), distance=model.SIM_SPLUS,k=600,alpha=0.25,beta=0.5,shrink=10,l=0.25,c=0.5)
    #     print('Saving the R^...')
    #     model.save_r_hat(evaluation=choice)
    # elif arg == 's':
    #     model.fit(data.get_urm_train_2(), distance=model.SIM_SPLUS,k=600,alpha=0.25,beta=0.5,shrink=10,l=0.25,c=0.5)
    #     print('Saving the similarity matrix...')
    #     sps.save_npz('raw_data/saved_sim_matrix_evaluation_2/{}'.format(model.name), model.get_sim_matrix())
    # elif arg == 'v':
    #     # model.validate(iterations=10, urm_train=data.get_urm_train_1(), urm_test=data.get_urm_test_1(), targetids=data.get_target_playlists(),
    #     #          distance=model.SIM_SPLUS, k=(100, 600), alpha=(0,2), beta=(0,2),shrink=(0,100),l=(0,1),c=(0,1))
    #     model.validate(iterations=10, urm_train=data.get_urm_train_1(), urm_test=data.get_urm_test_1(), targetids=data.get_target_playlists(),
    #              distance=model.SIM_RP3BETA, k=(100, 600), alpha=(0,2), beta=(0,2),shrink=(0,100),l=1,c=1)
    #     #model.test(distance=CFItemBased.SIM_P3ALPHA, k=300,alpha=(0,2),shrink=(0,100))
    # elif arg == 'x':
    #     pass
    # else:
    #     log.error('Wrong option!')

