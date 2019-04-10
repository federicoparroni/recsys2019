import sys
import os

import pandas as pd

sys.path.append(os.getcwd())

import data

from clusterize.clusterize_base import ClusterizeBase

class ClusterSessionsWithoutNumericalReferences(ClusterizeBase):

    def __init__(self):
        super(ClusterSessionsWithoutNumericalReferences, self).__init__()
        self.name = 'cluster_sessions_no_numerical_reference'

    def _fit(self, mode):
        """
        Cluster and predict for the test sessions without any numerical reference interactions

        self.train_indices: will contain all the train interactions
        self.test_indices: will contain all the test interactions
        self.target_indices: will contain the test interactions of sessions without
                            any other numerical reference interaction
        """
        # use all interactions for training
        self.train_indices = data.train_df(mode).index.values

        test_df = data.test_df(mode)
        self.test_indices = test_df.index.values

        # take only sessions without any numerical reference interactions except the clickout
        x = test_df.groupby(['session_id','user_id'])

        test_df = x.filter(lambda x: (x[x['action_type'] == 'clickout item'].shape[0] == 1) & (x[pd.to_numeric(x['reference'], errors='coerce').notnull()].shape[0] == 0))

        if test_df.shape[0] > 0:
            self.target_indices = test_df[test_df.action_type == 'clickout item'].index.values

        print(self.target_indices)

if __name__ == '__main__':
    import utils.menu as menu

    obj = ClusterSessionsWithoutNumericalReferences()
    
    mode = menu.mode_selection()
    
    obj.save(mode)
