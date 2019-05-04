import sys
import os
sys.path.append(os.getcwd())

from clusterize.clusterize_base import ClusterizeBase
import data
import utils.menu as menu
from tqdm.auto import tqdm
import numpy

class ClusterRecurrent(ClusterizeBase):

    """
    Cluster used to train and test the recurrent models, the dataset
    is trasformed into vectors.
    SESSIONS WITH NO CLICKOUTS or WITH A CLICKOUT REFERENCE NOT IN THE IMPRESSIONS ARE DISCARDED!
    Train: train sessions with at least one clickout
    Test: train sessions with at least one clickout
    Targets: all target
    """

    def __init__(self):
        super(ClusterRecurrent, self).__init__('cluster_recurrent')

    def _fit(self, mode):
        tqdm.pandas()
        def remove_bad_sessions(df):
            def filter_sessions(g):
                # check if there is at least 1 clickout
                clickouts = g[g.action_type == 'clickout item']
                if clickouts.shape[0] > 0:
                    # and the last clickout reference is in the impressions or is NaN (in test)
                    last_clickout = clickouts.iloc[[-1]]
                    imprs = last_clickout.impressions.str.split('|').values[0]
                    ref = last_clickout.reference
                    if ref.isnull().values[0] or ref.values[0] in imprs:
                        return g
            
            df = df.reset_index().groupby(['user_id','session_id']).progress_apply(filter_sessions)
            return df.set_index('index').sort_index()

        train_df = data.train_df(mode)
        train_df = remove_bad_sessions(train_df)
        self.train_indices = train_df.index.values
        del train_df

        test_df = data.test_df(mode)
        test_df = remove_bad_sessions(test_df)
        self.test_indices = test_df.index.values

        self.target_indices = []


if __name__ == '__main__':
    obj = ClusterRecurrent()
    
    mode = menu.mode_selection()
    obj.save(mode)
