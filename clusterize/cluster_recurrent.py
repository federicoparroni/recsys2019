import sys
import os
sys.path.append(os.getcwd())

from clusterize.clusterize_base import ClusterizeBase
import data

class ClusterRecurrent(ClusterizeBase):

    """
    Cluster used to train and test the recurrent models, the dataset
    is trasformed into vectors
    Train: full train
    Test: full test
    Targets: all target
    """

    def __init__(self):
        super(ClusterRecurrent, self).__init__('cluster_recurrent')

    def _fit(self, mode):
        self.train_indices = data.train_indices(mode)
        
        df = data.test_df(mode)
        self.test_indices = df.index.values

        self.target_indices = []


if __name__ == '__main__':
    obj = ClusterRecurrent()
    #obj.save('full')
    obj.save('local')
    #obj.save('small')
