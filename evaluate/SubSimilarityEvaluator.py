import utils.functions as f
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from prettytable import PrettyTable
import numpy as np
from scipy.stats import stats

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
    You need an evaluator folder:

        submissions
            |__evaluator

    In the evaluator folder, you need:
    - the first submission
    - the second submission
    """

    def __init__(self, sub1, sub2, distance='kendalltau'):
        # sub1 e sub2 sono i nomi delle sub, e devono essere nella cartella /similarity_evaluator
        self.current_directory = Path(__file__).absolute().parent
        self.data_directory = self.current_directory.joinpath('..', 'submissions/evaluator')
        self.distance = distance

        self.sub1 = pd.read_csv(self.data_directory.joinpath(sub1))
        self.sub2 = pd.read_csv(self.data_directory.joinpath(sub2))

        common_sessions = self.check_submissions()
        self.sub1 = self.sub1[self.sub1['session_id'].isin(common_sessions)]
        self.sub2 = self.sub2[self.sub2['session_id'].isin(common_sessions)]

        self.sub1 = self.sub1.sort_values(by=['user_id', 'timestamp'])
        self.sub1 = self.sub1.reset_index(drop=True)
        self.sub2 = self.sub2.sort_values(by=['user_id', 'timestamp'])
        self.sub2 = self.sub2.reset_index(drop=True)

    def check_submissions(self):
        print('Checking submissions...')
        sub1_sessions = self.sub1['session_id'].unique().tolist()
        sub2_sessions = self.sub2['session_id'].unique().tolist()
        common_sessions = set(sub1_sessions) & set(sub2_sessions)
        return common_sessions

    def run(self):
        print(f'\033[1;32;40m {cool_string}'+'\033[0;37;40m')
        # TODO: add self.generate_clusters() per valutare sui clusters?
        cumulative_distance = 0
        n_samples = self.sub1.shape[0]
        for i in tqdm(range(n_samples)):
            recommendations_1 = str(self.sub1.loc[i,'item_recommendations']).split()
            recommendations_1 = [int(x) for x in recommendations_1]
            recommendations_2 = str(self.sub2.loc[i,'item_recommendations']).split()
            recommendations_2 = [int(x) for x in recommendations_2]

            if len(recommendations_1)==len(recommendations_2):
                if len(recommendations_1)==1:
                    if recommendations_1[0] == recommendations_2[0]:
                        distance = 1
                    else:
                        distance = 0
                else:
                    distance= self.compute_distance(recommendations_1, recommendations_2)
                cumulative_distance = cumulative_distance + distance
        print(f"Kendall's Tau:{cumulative_distance/n_samples}")


    def compute_distance(self, recommendation_list_1, recommendation_list_2):
        if self.distance == 'kendalltau':
            distance, p_value = stats.kendalltau(recommendation_list_1, recommendation_list_2)
        if self.distance == 'weighted_kendalltau':
            distance, p_value = stats.weightedtau(recommendation_list_1, recommendation_list_2)
        return distance

if __name__=='__main__':
    sub_evaluator = SubSimilarityEvaluator('lazyUserRec.csv', 'xgboostlocal.csv')
    sub_evaluator.run()
