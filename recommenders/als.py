import numpy as np
import implicit
from recommenders.recommender_base import RecommenderBase
import scipy.sparse as sps
import data
from tqdm import tqdm
import os
import utils.check_folder as cf
from validator import BayesianValidator


class AlternatingLeastSquare(RecommenderBase):
    """
    Reference: http://yifanhu.net/PUB/collaborative_filtering.pdf (PAPER)
               https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe (SIMPLE EXPLANATION)

    Implementation of Alternating Least Squares with implicit data. We iteratively
    compute the user (x_u) and item (y_i) vectors using the following formulas:

    x_u = ((Y.T*Y + Y.T*(Cu - I) * Y) + lambda*I)^-1 * (X.T * Cu * p(u))
    y_i = ((X.T*X + X.T*(Ci - I) * X) + lambda*I)^-1 * (Y.T * Ci * p(i))

    [link text](http://www.example.com)
    """

    def __init__(self, mode, cluster, urm_name, factors=100, regularization=0.01, iterations=10, alpha=25):
        os.environ['MKL_NUM_THREADS'] = '1'
        name = 'ALS urm_name: {}\n factors: {}\n regularization: {}\n ' \
                    'iterations: {}\n alpha: {}'.format(urm_name, factors, regularization, iterations, alpha)
        super(AlternatingLeastSquare, self).__init__(mode, cluster, name)

        self.factors = int(factors)
        self.regularization = regularization
        self.iterations = int(iterations)
        self.alpha = int(alpha)

        self.target_indices = data.target_indices(mode, cluster)

        self.dict_row = data.dictionary_row(mode, cluster)
        self.target_indices_urm = []
        for ind in self.target_indices:
            self.target_indices_urm.append(self.dict_row[tuple(data.full_df().loc[ind][['session_id', 'user_id']])])


        self.urm = data.urm(mode=mode, cluster=cluster, urm_name=urm_name)
        self.user_vecs = None
        self.item_vecs = None
        self._model = None

        self.fixed_params_dict = {
            'mode': mode,
            'urm_name': urm_name,
            'cluster': cluster
        }

        self.hyperparameters_dict = {
            'factors': (50, 200),
            'regularization': (0, 1),
            'iterations': (1, 250),
            'alpha': (15, 45)
        }

    def get_r_hat(self):
        """
        compute the r_hat for the model filled with zeros in playlists not target
        :return  r_hat
    """
        print('computing the R_hat...')
        return self.user_vecs[self.target_indices_urm]*self.item_vecs.T

    def fit(self):
        """
        train the model finding the two matrices U and V: U*V.T=R  (R is the extimated URM)

        Parameters
        ----------
        :param (csr) urm: The URM matrix of shape (number_users, number_items).
        :param (int) factors: How many latent features we want to compute.
        :param (float) regularization: lambda_val regularization value
        :param (int) iterations: How many times we alternate between fixing and updating our user and item vectors
        :param (int) alpha: The rate in which we'll increase our confidence in a preference with more interactions.

        Returns
        -------
        :return (csr_matrix) user_vecs: matrix N_user x factors
        :return (csr_matrix) item_vecs: matrix N_item x factors
        """

        sparse_item_user = self.urm.T

        # Initialize the als model and fit it using the sparse item-user matrix
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        self._model = implicit.als.AlternatingLeastSquares(factors=self.factors,
                                                           regularization=self.regularization,
                                                           iterations=self.iterations)

        # Calculate the confidence by multiplying it by our alpha value.
        data_conf = (sparse_item_user * self.alpha).astype('double')

        # Fit the model
        self._model.fit(data_conf)

        # set the user and item vectors for our model R = user_vecs * item_vecs.T
        self.user_vecs = self._model.user_factors[self.target_indices_urm]
        self.item_vecs = self._model.item_factors

    def recommend_batch(self):
        print('recommending batch')

        full_df = data.full_df()
        dict_col = data.dictionary_col(mode=self.mode)

        predictions = []

        count = 0
        for index in tqdm(self.target_indices):
            impr = list(map(int, full_df.loc[index]['impressions'].split('|')))
            columns = [dict_col[i] for i in impr]
            item_vecs = self.item_vecs[columns]
            r_hat_row = np.dot(self.user_vecs[count], item_vecs.T)
            l = list(zip(impr, r_hat_row.tolist()))
            l.sort(key=lambda tup: tup[1], reverse=True)
            p = [e[0] for e in l]
            predictions.append((index, p))
            count += \
                1
        print('PRED CREATED')
        return predictions

    def save_r_hat(self):
        base_save_path = 'dataset/matrices/{}/r_hat_matrices'.format(self.mode)
        cf.check_folder(base_save_path)
        print('saving r_hat...')
        sps.save_npz('{}/{}'.format(base_save_path, self.name), self.get_r_hat())
        print('r_hat saved succesfully !')

    def get_scores_batch(self):
        print('to be implemented')
        pass


if __name__ == '__main__':
    model = AlternatingLeastSquare(mode='local', cluster='no_cluster', urm_name='urm_session_aware_lin', factors=100, regularization=0.05,
                                   iterations=100, alpha=25)
    val = BayesianValidator(model)
    val.validate(iterations=10)

