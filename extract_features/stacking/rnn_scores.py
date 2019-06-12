import pandas as pd
import numpy as np
import data
import os
import glob
from preprocess_utils.extract_scores import *
from extract_features.feature_base import FeatureBase
from preprocess_utils.last_clickout_indices import find
from pathlib import Path


class ScoresRNN(FeatureBase):

    """
        RNN scores
        Se vogliamo metterlo in una sottocartella di extract feature, cambiare il name con
    """

    def __init__(self, mode, cluster='no_cluster'):
        name = 'scoresrnn'
        super(ScoresRNN, self).__init__(name=name, mode=mode, cluster=cluster)

    def extract_feature(self):
        self.current_directory = Path(__file__).absolute().parent
        self.data_dir = self.current_directory.joinpath('..', '..', 'stacking', self.mode)

        train = data.train_df(mode=self.mode, cluster=self.cluster)
        test = data.test_df(mode=self.mode, cluster=self.cluster)
        df = pd.concat([train, test])
        last_indices = find(df)

        # extract scores
        self.train_dir = self.data_dir.joinpath('test')
        for file in glob.glob(str(self.train_dir) + '/rnn*'):
            rnn = np.load(file)
            rnn = pd.DataFrame(rnn, columns=['index', 'item_recommendations', 'scores'])
            rnn = rnn.astype({'index': int})
            rnn = rnn[rnn['index'].isin(last_indices)]


        rnn_idx = list(rnn['index'])
        print(f'Rnn indices are : {len(set(rnn_idx))}')
        print(f'Last indices are : {len((last_indices))}')
        common = set(rnn_idx) & set(last_indices)
        print(f'In common : {len(common)}')

        t = assign_score(rnn, 'rnn')
        t = t.sort_values(by='index')

        df['index'] = df.index.values
        df = df[['user_id', 'session_id', 'index']]
        df = pd.merge(t, df, how='left', on=['index'])
        num_idx = len(set(df['index'].values))
        print(num_idx)
        return df[['user_id', 'session_id', 'item_id', 'score_rnn']]




if __name__ == '__main__':

    c = ScoresRNN(mode='local', cluster='no_cluster')

    c.extract_feature()


