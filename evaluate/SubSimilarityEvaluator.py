import utils.functions as f
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from prettytable import PrettyTable

import scipy.stats as stats

# stringa fatta su http://patorjk.com/software/taag/#p=display&f=Standard&t=Sub%20Evaluator%0A (font Standard)
cool_string = """
  ____  _           _ _            _ _           _____            _             _
 / ___|(_)_ __ ___ (_) | __ _ _ __(_) |_ _   _  | ____|_   ____ _| |_   _  __ _| |_ ___  _ __
 \___ \| | '_ ` _ \| | |/ _` | '__| | __| | | | |  _| \ \ / / _` | | | | |/ _` | __/ _ \| '__|
  ___) | | | | | | | | | (_| | |  | | |_| |_| | | |___ \ V / (_| | | |_| | (_| | || (_) | |
 |____/|_|_| |_| |_|_|_|\__,_|_|  |_|\__|\__, | |_____| \_/ \__,_|_|\__,_|\__,_|\__\___/|_|
                                         |___/
"""


class SubSimilarityEvaluator():
    """
    Evaluator of the similarity between submissions.
    You need an evaluator folder in similarity_evaluator:

        submissions
            |__similarity_evaluator

    In the evaluator folder, you need:
    - the first submission
    - the secondo submission
    - the test set (local for local sub evaluator, full for full sub evaluator)
    """

    def __init__(self, sub1, sub2, distance='kendalltau'):
        # sub1 e sub2 sono i nomi delle sub, e devono essere nella cartella /similarity_evaluator
        self.current_directory = Path(__file__).absolute().parent
        self.data_directory = self.current_directory.joinpath('..', 'submissions/similarity_evaluator')
        self.gt_csv = self.data_directory.joinpath('test.csv')
        self.distance = distance
        common_sessions = self.check_submissions()
        num_file = 0
        self.sub1 = pd.read_csv(self.data_directory.joinpath(sub1))
        self.sub2 = pd.read_csv(self.data_directory.joinpath(sub2))

        self.sub1 = self.sub1[self.sub1['session_id'].isin(common_sessions)]
        self.sub2 = self.sub2[self.sub2['session_id'].isin(common_sessions)]

        self.sub1 = self.sub1.sort_values(by=['user_id', 'timestamp'])
        self.sub1 = self.sub1.reset_index(drop=True)
        self.sub2 = self.sub2.sort_values(by=['user_id', 'timestamp'])
        self.sub2 = self.sub2.reset_index(drop=True)

    def check_submissions(self):
        print('Checking submissions...')
        df= pd.read_csv(self.gt_csv)
        sessions = df['session_id'].unique().tolist()
        for root, dirs, files in os.walk(self.data_directory):
            for file in files:
                if file.endswith(".csv"):
                    tmp = pd.read_csv(self.data_directory.joinpath(file))
                    tmp_sessions = tmp['session_id'].unique().tolist()
                    common_sessions = set(sessions) & set(tmp_sessions)
                    sessions = list(common_sessions)
        return sessions

    def run(self):
        print(f'\033[33;40m {cool_string}'+'\033[0;37;40m')
        # TODO: add self.generate_clusters() per valutare sui clusters?
        for i in range(5):
            recommendations_1 = str(self.sub1.loc[i, 'item_recommendations']).split()
            recommendations_2 = str(self.sub2.loc[i, 'item_recommendations']).split()
            print(recommendations_1)
            print(recommendations_2)
            distance = self.compute_distance(recommendations_1, recommendations_2)
            print(distance)


    def compute_distance(self, recommendation_list_1, recommendation_list_2):
        if self.distance == 'kendalltau':
            distance, p_value = stats.kendalltau(recommendation_list_1, recommendation_list_2)
        return distance

if __name__=='__main__':
    sub_evaluator = SubSimilarityEvaluator('lazyUserRec.csv', 'xgboostlocal.csv')
    sub_evaluator.run()
