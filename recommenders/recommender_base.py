from abc import abstractmethod
from abc import ABC
import data
import numpy as np
from tqdm import tqdm
import out


class RecommenderBase(ABC):
    """ Defines the interface that all recommendations models expose """

    def __init__(self, mode='full', name='recommenderbase'):
        """
        init should have on its firm the params of the algorithm
        """
        self.name = name
        self.mode = mode

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
        [(session_id_0, [acc_1, acc2, acc3, ...]), 
         (session_id_1, [acc_1, acc2, acc3, ...]), ...]
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
            out.create_sub(recommendations, mode=mode, submission_name=self.name)

    def evaluate(self):
        """
        Validate the model on local data
        """
        assert self.mode == 'local' or self.mode == 'small'

        print('\nvalidating {}'.format(self.name))
        self.fit()
        recommendations = self.recommend_batch()
        return self.compute_MRR(recommendations)

    def compute_MRR(self, predictions):
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

        for key, value in tqdm(predictions.items()):
            target_mask = handle["session_id"] == key
            target_reference = handle[target_mask]
            target_reference = list(target_reference["reference"])[0]
            RR += 1 / (value.index(target_reference) + 1)

        print("MRR is: {}".format(RR / target_session_count))

        return RR / target_session_count

    def get_params(self):
        # needed an override on each class
        pass
