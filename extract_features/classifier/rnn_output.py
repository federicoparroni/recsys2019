import math
from numpy import mean

from extract_features.feature_base import FeatureBase
import data
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()


class RNNOutput(FeatureBase):


    def __init__(self, mode, cluster='no_cluster'):
        name = 'rnn_output'
        super(RNNOutput, self).__init__(
            name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        pass


if __name__ == '__main__':
    from utils.menu import mode_selection

    mode = mode_selection()
    c = RNNOutput(mode=mode, cluster='no_cluster')
    c.save_feature()
