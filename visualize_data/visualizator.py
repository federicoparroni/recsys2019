import os
import sys
sys.path.append(os.getcwd())

import data
import pandas as pd
import matplotlib.pyplot as plt


class visualizator():
    def __init__(self):
        self.train_df = data.train_df('full')
        self.test_df = data.test_df('full')
        self.complete_df = pd.concat([self.train_df, self.test_df])

    def visualize_sessions_length(self):

        df_test = self.test_df[['session_id', 'action_type']].groupby('session_id').agg('count')
        df_train = self.train_df[['session_id', 'action_type']].groupby('session_id').agg('count')
        df_complete = self.complete_df[['session_id', 'action_type']].groupby('session_id').agg('count')

        df_train.hist(bins='auto')
        plt.show(block=True)

        df_test.hist(bins='auto')
        plt.show(block=True)

        df_complete.hist(bins='auto')
        plt.show(block=True)


def visualize_session_per_user(df):
  pass


if __name__ == "__main__":
    visualizator = visualizator()
    visualizator.visualize_sessions_length()