from abc import abstractmethod
from abc import ABC

class RecommenderBase(ABC):
    """ Defines the interface that all recommendations models expose """

    def __init__(self):
        self.name = 'recommenderbase'

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

    def recommend_batch(self, userids, urm,  N=10, filter_already_liked=True, with_scores=True, items_to_exclude=[], verbose=False):
        pass

    def evaluate(self, recommendations, test_urm, at_k=10, single_ap=False, verbose=True):
        pass
