from recommenders.recommender_base import RecommenderBase
import scipy.sparse as sps
import utils.log as log
import time
from bayes_opt import BayesianOptimization
import sklearn.preprocessing as sk
import utils.check_matrix_format as cm
import data
from tqdm import tqdm


class Hybrid(RecommenderBase):
    """
    recommender builded passing to the init method an array of (matrices_array) that will be combined to obtain an hybrid r_hat
    """
    MAX_MATRIX = 'MAX_MATRIX'
    MAX_ROW = 'MAX_ROW'
    L2 = 'L2'
    NONE = 'NONE'

    def __init__(self, name, cluster, mode, matrices_array, normalization_mode, weights_array):

        super(Hybrid, self).__init__(name=name, cluster=cluster, mode=mode)

        # load handle and dictionary based on mode will be used during the recommend batch
        self.dict_col = data.dictionary_col(mode=self.mode)
        self.df_handle = data.handle_df(mode=self.mode)
        self.targetids = data.target_urm_rows(self.mode)
        self.r_hat = None

        # will be set if the hybrid is done via similarity matrices
        self.urm_name = None
        self.weights_array = weights_array

        # store the array of matrices in the hybrid recommender
        self.matrices_array = matrices_array

        # check the shapes of the matrices
        self._check_matrices_array_shapes()

        # normalize the matrices
        self.normalization_mode = normalization_mode

        # will be filled when the _normalization method will be called
        self.normalized_matrices_array = None

        print('checking that all the matrix in matrices array are in CSR format...\n')
        for index in range(len(self.matrices_array)):
            self.matrices_array[index] = cm.check_matrix(self.matrices_array[index], 'csr')
        print('done\n')

        print('normalizing the matrix in matrices array...\n')
        self._normalization(normalization_mode=self.normalization_mode)
        print('matrices_normalized\n')

    def fit(self):
        print('hybrid matrix creation...')
        start = time.time()
        hybrid_matrix = sps.csr_matrix(self.normalized_matrices_array[0].shape)

        count = 0
        for m in self.normalized_matrices_array:
            hybrid_matrix += m * self.weights_array[count]
            count += 1

        if self.name == 'HybridSimilarity':
            # compute the r_hat if we have the similarity
            urm = data.urm(self.mode, self.urm_name)
            # check that urm is in CSR format
            urm = cm.check_matrix(urm, 'csr')
            # check if the similarity is user-user or item-item
            if hybrid_matrix.shape[0] == urm.shape[1]:
                # user - user similarity
                hybrid_matrix = urm[self.targetids].dot(hybrid_matrix)
            else:
                # item - item similarity
                hybrid_matrix = hybrid_matrix[self.targetids].dot(urm)
        print('hybrid matrix created in {:.2f} s'.format(time.time() - start))
        self.r_hat = hybrid_matrix

    def get_r_hat(self):
        self._has_fit()
        return self.r_hat

    def recommend_batch(self):
        # check if the model has fit
        self._has_fit()
        print('computing predictions...')
        start = time.time()
        predictions = []
        for index, row in tqdm(self.df_handle.iterrows()):
            impressions = list(map(int, row['impressions'].split('|')))
            impressions_score_tuples = [[i, self.r_hat[index, self.dict_col[i]]] for i in impressions]
            impressions_score_tuples.sort(key=lambda tup: tup[1], reverse=True)
            predictions.append((row['session_id'], [e[0] for e in impressions_score_tuples]))

        print('predictions computed in {:.2f} s'.format(time.time() - start))
        return predictions

    def _check_matrices_array_shapes(self):
        shape = self.matrices_array[0].shape
        for m in self.matrices_array:
            if m.shape != shape:
                print("the matrices passed have not the same shape... go get some coffe...")
                exit(0)

    def _normalize_max_row(self):
        """
        :param userids: user for which compute the predictions
        :return: return an array containing the normalized matrices
        """
        normalized_matrices_array = []
        count = 0
        for r in self.matrices_array:
            for row_index in range(r.shape[0]):
                print(row_index * 100 / r.shape[0])

                row = r.getrow(row_index)
                max_row = row.max()
                r[row_index].data = r[row_index].data / max_row

                # row = r[row_index]
                # max_row = row.max()
                # normalized_row = row/max_row
                # r[row_index] = normalized_row
            normalized_matrices_array.append(r)
            count += 1
        return normalized_matrices_array

    def _normalize_max_matrix(self):
        """
        :param userids: user for which compute the predictions
        :return: return an array containing the normalized matrices
        """
        normalized_matrices_array = []
        count = 0
        for r in self.matrices_array:
            # let the values positives
            # min = r.min()
            # print(min)
            # if min < 0:
            #    r.data = r.data - min

            # avg = np.sum(r.data) / len(r.data)
            # print('avg: {}'.format(avg))
            # r.data = r.data - avg

            max_matrix = r.max()
            print('max: {}'.format(max_matrix))

            # print('avg/max: {}'.format(avg/max_matrix))

            # r.data = r.data*(avg/max_matrix)

            r.data = r.data / max_matrix

            # adding confidence
            # r.data=r.data*(r.data-avg)*100
            # for ind in range(len(r.data)):
            #    confidence = (r.data[ind]-avg)*100
            #    r.data[ind] = r.data[ind]*confidence

            normalized_matrices_array.append(r)
            count += 1
        return normalized_matrices_array

    def _normalize_l2(self):
        normalized_matrices_array = []
        for r in self.matrices_array:
            r = sk.normalize(r)
            normalized_matrices_array.append(r)
        return normalized_matrices_array

    def _normalization(self, normalization_mode):
        if normalization_mode == 'MAX_ROW':
            self.normalized_matrices_array = self._normalize_max_row()
        elif normalization_mode == 'MAX_MATRIX':
            self.normalized_matrices_array = self._normalize_max_matrix()
        elif normalization_mode == 'NONE':
            self.normalized_matrices_array = self.matrices_array
        elif normalization_mode == 'L2':
            self.normalized_matrices_array = self._normalize_l2()
        else:
            log.error('invalid string for normalization')
            return

    def _has_fit(self):
        if self.r_hat is None:
            self.fit()

    # TODO: reimplemented bayesian search validation
    """
    def validateStep(self, **dict):
        # gather saved parameters from self
        targetids = self._validation_dict['targetids']
        urm_test = self._validation_dict['urm_test']
        N = self._validation_dict['N']
        filter_already_liked = self._validation_dict['filter_already_liked']
        items_to_exclude = self._validation_dict['items_to_exclude']

        # build weights array from dictionary
        weights = []
        for i in range(len(dict)):
            w = dict['w{}'.format(i)]
            weights.append(w)

        # evaluate the model with the current weigths
        recs = self.recommend_batch(weights, target_userids=targetids, N=N,
                                    filter_already_liked=filter_already_liked, items_to_exclude=items_to_exclude,
                                    verbose=False)
        return self.compute_MRR(recs, test_urm=urm_test)

    def validate(self, iterations, urm_test, userids=None,
                 N=10, filter_already_liked=True, items_to_exclude=[], verbose=False):
        # save the params in self to collect them later
        self._validation_dict = {
            'targetids': userids,
            'urm_test': urm_test,
            'N': N,
            'filter_already_liked': filter_already_liked,
            'items_to_exclude': items_to_exclude
        }

        pbounds = {}
        for i in range(len(self.matrices_array)):
            pbounds['w{}'.format(i)] = (0, 1)

        optimizer = BayesianOptimization(
            f=self.validateStep,
            pbounds=pbounds,
            random_state=1,
        )
        optimizer.maximize(
            init_points=2,
            n_iter=iterations,
        )

        print(optimizer.max)
        return optimizer

    def run(self):
        pass
    """

