from abc import abstractmethod
from abc import ABC

class ClusterizeBase(ABC):

    def __init__(self, name='clusterbase'):
        self.name = name

    @abstractmethod
    def fit(self):
        """
        should return a tuple:
        (:,,) -> indices of base_split/full/train to be included in cluster full training set
        (,:,) -> indices of base_split/full/test  to be included in cluster full test set
        (,,:) -> indices of base_split/full/test  that represents clickouts to be predicted
        
        third argument is optional: one should specify it only in case wants to impose the clickouts
                                    to be predicted. otherwise those will be set automatically equal
                                    to the missing clickouts indicated by the 2nd argument 
        """
        pass
    
    def save():
        """
        makes use of fit to create the dataset for a specific cluster. in particular it take cares
        to create a folder at the same level of base_split with the specified name and the structure
        of folders inside 
        """
        