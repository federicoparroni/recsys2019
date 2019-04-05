import sys
import os
sys.path.append(os.getcwd())

import data
import numpy as np
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
        test_df = test_df.groupby(['session_id','user_id'], group_keys=False)\
            .filter(lambda g: g[(g.action_type != 'clickout_item') & g.reference.fillna('').str.isnumeric()].shape[0] == 0 )
        
        if test_df.shape[0] > 0:
            self.target_indices = test_df[test_df.action_type == 'clickout_item'].index.values
    

if __name__ == '__main__':
    import utils.menu as menu

    obj = ClusterSessionsWithoutNumericalReferences()
    
    mode = menu.mode_selection()
    
    obj.save(mode)
