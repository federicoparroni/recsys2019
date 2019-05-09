import sys
import os
sys.path.append(os.getcwd())

from clusterize.clusterize_base import ClusterizeBase
import data
import utils.menu as menu
from tqdm.auto import tqdm
import numpy

class ClusterOverLen6(ClusterizeBase):

    """
    Cluster used to train and test the recurrent models, the dataset
    is trasformed into vectors.
    SESSIONS WITH NO CLICKOUTS or HAVING LESS THAN 7 INTERACTION ARE DISCARDED!
    Train: train sessions with at least one clickout and grater than 6 interactions
    Test: train sessions with at least one clickout and greater than 6 interactions
    Targets: all target
    """

    def __init__(self):
        super(ClusterOverLen6, self).__init__('cluster_over_len6')

    def _fit(self, mode):
        tqdm.pandas()
        def filter_sessions(df):
            def filter_fn(g):
                # check if there is at least 1 clickout
                clickouts = g[g.action_type == 'clickout item']
                return len(clickouts) > 0 and len(g) > 6

            return df.groupby(['user_id','session_id']).filter(filter_fn)

        train_df = data.train_df(mode)
        train_df = filter_sessions(train_df)
        self.train_indices = train_df.index.values
        del train_df

        test_df = data.test_df(mode)
        test_df = filter_sessions(test_df)
        self.test_indices = test_df.index.values

        self.target_indices = []


if __name__ == '__main__':
    c = ClusterOverLen6()
    
    mode = menu.mode_selection()
    c.save(mode)
