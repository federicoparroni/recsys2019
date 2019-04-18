from abc import abstractmethod
from abc import ABC
from utils.check_folder import check_folder
import pandas as pd
from utils.menu import yesno_choice
import os

"""
extend this class and give an implementation to extract_feature to 
make available a new feature
"""


class FeatureBase(ABC):

    """
    columns_to_onehot: [(columns_header, onehot_mode), ...]
        onehot_mode: 'single' or 'multiple'
            'single': if we have just one categorical value for row
            'multiple': if we have multiple ones (assumes pipe separation)

    eg: [('action', 'single')]
    meaning that the header of the column to onehot is 'action' and the onehot modality is 'single'
    """

    def __init__(self, mode, cluster='no_cluster', name='featurebase', columns_to_onehot=[]):
        self.mode = mode
        self.cluster = cluster
        self.name = name
        self.columns_to_onehot = columns_to_onehot

    """
    returns a dataframe that contains a feature (or more than one)
    on the first columns it should have an identifier of the single object to which the feature refers
    on the other column (or columns), the value of the features, with a meaningful name for the header.

    in case of categorical features, DO NOT RETURN A ONEHOT!
    In particular, return a single categorical value or a list of pipe-separated categorical values, and
    take care of setting self.categorical_variables nicely: base class will take care of one honetizing
    when read_feature is called.
    """
    @abstractmethod
    def extract_feature(self):
        pass

    def save_feature(self):
        path = 'dataset/preprocessed/{}/{}/feature/{}/features.csv'.format(
            self.cluster, self.mode, self.name)
        if os.path.exists(path):
            choice = yesno_choice('feature exists yet. want to recreate?')
            if choice == 'n':
                return
        df = self.extract_feature()
        check_folder(path)
        df.to_csv(path)

    def read_feature(self, one_hot=False):
        path = 'dataset/preprocessed/{}/{}/feature/{}/features.csv'.format(
            self.cluster, self.mode, self.name)
        if not os.path.exists(path):
            choice = yesno_choice('feature does not exist. want to create?')
            if choice == 'y':
                self.save_feature()
            else:
                return

        df = pd.read_csv(path)
        df = df.drop('Unnamed: 0', axis=1)

        print('{} feature read'.format(self.name))

        # then proceed with one hot
        if one_hot:
            for t in self.columns_to_onehot:
                col = df[t[0]]
                if t[1] == 'single':
                    oh = pd.get_dummies(col)
                else:
                    oh = col.str.split(
                        '|', expand=True).stack().str.get_dummies().sum(level=0)
                df = df.drop([t[0]], axis=1)
                df = df.join(one_hot)
            
            print('{} onehot completed'.format(self.name))

        return df

