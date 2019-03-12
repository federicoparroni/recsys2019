from abc import abstractmethod
from abc import ABC
import data
import numpy as np
from tqdm import tqdm


class RecommenderBase(ABC):
    """ Defines the interface that all recommendations models expose """

    def __init__(self, mode='full', urm_name='urm_clickout'):
        self.name = 'recommenderbase'
        self.mode = mode
        self.urm_name = urm_name

    @abstractmethod
    def fit(self):
        """
        Fit the model on the data. Inherited class should extend this method in the appropriate way.
        """
        pass

    @abstractmethod
    def run(self):
        """
        Handle all the operations needed to run this model a single time.
        In particular, creates the object, performs the fit and get the recommendations.
        Then, it can either evaluate the recommendations or export the model
        """
        pass

    def recommend_batch(self, df_handle, dict):
        pass

    def evaluate(self, predictions, mode, verboose=True):
        """
        compute the MRR mean reciprocal rank of some predictions
        it uses the mode parameter to know which handle to retrieve to compute the score

        :param mode: 'local' or 'small' say which train has been used
        :param predictions: session_id, ordered impressions_list
        :param verboose: if True print the MRR
        :return: MRR of the given predictions
        """
        assert (mode == 'local' or mode == 'small')

        handle = data.handle_df(mode)
        test = np.array(handle)

        # initialize reciprocal rank value
        RR = 0

        target_session_count = test.shape[0]

        for i in tqdm(range(target_session_count)):
            position = (np.where(np.array(predictions[i][1]) == test[i, 4]))[0][0]
            RR += 1 / (position + 1)

        if verboose: print("MRR is: {}".format(RR / target_session_count))

        return RR / target_session_count
