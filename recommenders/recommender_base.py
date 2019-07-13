from abc import abstractmethod
from abc import ABC
import data
import numpy as np
from tqdm import tqdm
import out
import utils.telegram_bot as HERA
from utils.check_folder import check_folder
import time


class RecommenderBase(ABC):
    """ Defines the interface that all recommendations models expose """

    def __init__(self, mode='full', cluster='no_cluster', name='recommenderbase'):
        """
        init should have on its firm the params of the algorithm
        """
        assert mode in ['full', 'local', 'small']
        
        self.name = name
        self.mode = mode
        self.cluster = cluster
        self.weight_per_position = None

    @abstractmethod
    def fit(self):
        """
        Fit the model on the data. Inherited class should extend this method in the appropriate way.
        """
        pass

    @abstractmethod
    def recommend_batch(self):
        """
        Returns a list of recommendations in the format
        [(session_idx_0, [acc_1, acc2, acc3, ...]), 
         (session_idx_1, [acc_1, acc2, acc3, ...]), ...]
        """
        pass

    @abstractmethod
    def get_scores_batch(self):
        """
        returns a list of recommendations in the format
        [(session_idx_0, [acc_1, acc2, acc3, ...], [sco_1, sco2, sco3, ...]),
         (session_idx_0, [acc_1, acc2, acc3, ...], [sco_1, sco2, sco3, ...]), ...]
        """
        pass

    def set_weight_per_position(self, list_weight):
        """
        Set list values for weight_per_position parameter of recommenders.
        It is used for hybridation purposes.
        It is None when scores per position are given by a similarity matrix
        (such as what happens with DistanceBasedRecommenders.
        """
        if len(list_weight)>25:
            print("The list has lenght > 25. It will be cut")
        self.weight_per_position = list_weight[:25]

    def run(self, export_sub=True, export_scores=False, evaluate=False):
        """
        Handle all the operations needed to run this model a single time.
        In particular, performs the fit and get the recommendations.
        Then, it can either export the submission or not based the flag export_sub.
        Moreover, it can export the scores of the algorithm based on the flag export_scores
        """
        print('running {}'.format(self.name))
        if export_scores:
            if export_sub:
                print("I gonna fit the model, recommend the accomodations, save the scores and export the submission")
            else:
                print("I gonna fit the model, recommend the accomodations and save the scores")
        else:
            if export_sub:
                print("I gonna fit the model, recommend the accomodations and export the submission")
            else:
                print("I gonna fit the model and recommend the accomodations")
        self.fit()

        if export_sub:
            recommendations = self.recommend_batch()
            out.create_sub(recommendations, submission_name=self.name)
        if export_scores:
            check_folder('scores')
            scores_batch_test = self.get_scores_batch()
            path = 'scores/{}_test_{}'.format(self.name, time.strftime('%H-%M-%S'))
            np.save(path, scores_batch_test)
            print('scores exported in {}'.format(path))
        if evaluate:
            recommendations = self.recommend_batch()
            print('recommendations created')
            MRR = self.compute_MRR(recommendations)
            HERA.send_message(
                'evaluating recommender {} on {}.\n MRR is: {}\n\n'.format(self.name, self.cluster, MRR))

    def evaluate(self, send_MRR_on_telegram = False, already_fitted=False):
        """
        Validate the model on local data
        """

        print('\nevaluating {}'.format(self.name))
        
        # infos on the perc of target indices in which I'm evaluating the model
        perc = len(data.target_indices(self.mode, self.cluster))/len(data.target_indices(self.mode, data.SPLIT_USED))
        print('\nevaluating with mode {} on {} percent of the targets\n'.format(self.mode, perc*100))

        if not already_fitted:
            self.fit()
        recommendations = self.recommend_batch()
        print('recommendations created')
        MRR = self.compute_MRR(recommendations)
        if send_MRR_on_telegram:
            HERA.send_message('evaluating recommender {} on {}.\n MRR is: {}\n\n'.format(self.name, self.cluster, MRR))
        return MRR

    def compute_MRR(self, predictions):
        """
        compute the MRR mean reciprocal rank of some predictions
        it uses the mode parameter to know which handle to retrieve to compute the score

        :param mode: 'local' or 'small' say which train has been used
        :param predictions: session_id, ordered impressions_list
        :param verboose: if True print the MRR
        :return: MRR of the given predictions
        """

        if self.mode == 'full':
            train_df = data.full_df()
        else:
            train_df = data.train_df('full')

        target_indices, recs = zip(*predictions)
        target_indices = list(target_indices)
        correct_clickouts = train_df.loc[target_indices].reference.values
        len_rec = len(recs)
        
        RR = 0
        print("Calculating MRR (hoping for a 0.99)")
        for i in tqdm(range(len_rec)):
            correct_clickout = int(correct_clickouts[i])
            if correct_clickout in predictions[i][1]:
                rank_pos = recs[i].index(correct_clickout) + 1
                if rank_pos <= 25:
                    RR += 1 / rank_pos
        
        MRR = RR / len_rec
        print(f'MRR: {MRR}')

        return MRR

    def get_params(self):
        """
        returns the dictionaries used for the bayesian search validation
        the two dictionaries have to be implemented by each recommenders in the init!

        """
        if (self.fixed_params_dict is None) or (self.hyperparameters_dict is None):
            print('dictionaries of the parameters have not been set on the recommender!!')
            exit(0)
        return self.fixed_params_dict, self.hyperparameters_dict

