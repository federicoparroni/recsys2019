from abc import abstractmethod
from abc import ABC
import data
import numpy as np
from tqdm import tqdm


class RecommenderBase(ABC):
    """ Defines the interface that all recommendations models expose """

    def __init__(self, mode='full', urm_name='urm_clickout'):
        """
        init should have on its firm the params of the algorithm
        """
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
    def recommend_batch(self):
        """
        returns a list of recommendations in the format
        [session_id_0 [acc_1, acc2, acc3, ...], 
         session_id_1 [acc_1, acc2, acc3, ...], ...]
        """
        pass

    def run(self):
        """
        Handle all the operations needed to run this model a single time.
        In particular, performs the fit and get the recommendations.
        Then, it can either export the recommendations or not
        """
        export = False
        print('running {}'.format(self.name))
        if self.mode == 'full':
            export = True
            print("I gonna fit the model, recommend the accomodations, and save the submission")
        else:
            print("I gonna fit the model and recommend the accomodations")

        self.fit()
        recommendations = self.recommend_batch()
        if export:
            out.create_sub(recommendations)

    def validate(self):
        """
        used to validate the model on local data
        """
        assert self.mode == 'local' or self.mode == 'small'

        print('\nvalidating {}'.format(self.name))
        self.fit()
        recommendations = self.recommend_batch()
        self.evaluate(recommendations)

    def evaluate(self, predictions):
        """
        compute the MRR mean reciprocal rank of some predictions
        it uses the mode parameter to know which handle to retrieve to compute the score

        :param mode: 'local' or 'small' say which train has been used
        :param predictions: session_id, ordered impressions_list
        :param verboose: if True print the MRR
        :return: MRR of the given predictions
        """
        assert (self.mode == 'local' or self.mode == 'small')

        handle = data.handle_df(self.mode)
        test = np.array(handle)

        # initialize reciprocal rank value
        RR = 0

        target_session_count = test.shape[0]

        for i in tqdm(range(target_session_count)):
            position = (np.where(np.array(predictions[i][1]) == test[i, 4]))[0][0]
            RR += 1 / (position + 1)

        print("MRR is: {}".format(RR / target_session_count))

        return RR / target_session_count
