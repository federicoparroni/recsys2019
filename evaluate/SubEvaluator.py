import utils.functions as f
import os

import data
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from prettytable import PrettyTable


# stringa fatta su http://patorjk.com/software/taag/#p=display&f=Standard&t=Sub%20Evaluator%0A (font Standard)
cool_string = """
  ____        _       _____            _             _
 / ___| _   _| |__   | ____|_   ____ _| |_   _  __ _| |_ ___  _ __
 \___ \| | | | '_ \  |  _| \ \ / / _` | | | | |/ _` | __/ _ \| '__|
  ___) | |_| | |_) | | |___ \ V / (_| | | |_| | (_| | || (_) | |
 |____/ \__,_|_.__/  |_____| \_/ \__,_|_|\__,_|\__,_|\__\___/|_|

"""


class SubEvaluator():
    """
    Evaluator of a local submission.
    You need an evaluator folder in submissions:

        submissions
            |__evaluator

    In the evaluator folder, you need:
    - the submission
    - the test set
    - the ground truth file (test with the missing clickouts)
    """

    def __init__(self, sub):
        self.current_directory = Path(__file__).absolute().parent
        self.data_directory = self.current_directory.joinpath('..', 'submissions/evaluator')
        self.sub = sub

    def score_sub(self, sub_path, gt_csv, total_score=-1):
        cluster_name = os.path.basename(self.data_directory.joinpath(gt_csv))
        cluster_name = os.path.splitext(cluster_name)[0]
        print("Computing score for"+f"\033[1;35;40m {sub_path}"+ '\033[0;37;40m' +" with cluster" + f"\033[1;35;40m {cluster_name}"+ '\033[0;37;40m')
        subm_csv = self.data_directory.joinpath(sub_path)
        mrr = f.score_submissions(subm_csv, gt_csv, f.get_reciprocal_ranks)
        if total_score != -1:
            if mrr > total_score:
                print('\033[1;40m Score: '+ f'\033[1;32;40m {mrr}'+ '\033[1;40m    :)' + '\033[0;37;40m')
            elif mrr < total_score-0.02 and mrr > total_score-0.04:
                print('\033[1;40m Score: '+ f'\033[1;31;40m {mrr}'+ '\033[1;40m    :(' + '\033[0;37;40m')
            elif mrr < total_score - 0.04:
                print('\033[1;40m Score: '+ f'\033[1;31;40m {mrr}'+ '\033[1;40m    ¯\_(⊙︿⊙)_/¯     <--------- PROBLEM HERE!' + '\033[0;37;40m')
            else:
                print('\033[1;40m Score: '+ f'\033[1;33;40m {mrr}'+ '\033[1;40m     :|' + '\033[0;37;40m')
        else:
            print('\033[1;40m Score: '+ f'\033[1;32;40m {mrr}'+ '\033[0;37;40m')
        return mrr

    def run(self, save_path=None):
        self.generate_clusters()
        print(f'\033[33;40m {cool_string}'+'\033[0;37;40m')
        gt_list = [
        'gt_only_mobile.csv',
        'gt_only_desktop.csv',
        'gt_only_one_interaction_sessions.csv',
        'gt_only_two_interaction_sessions.csv',
        'gt_less_than_5_interaction_sessions.csv',
        'gt_5_to_10_interaction_sessions.csv',
        'gt_more_than_10_interaction_sessions.csv',
        #TODO ADD OTHER CLUSTERS
        'gt_no_num_reference_sessions.csv'
        ]

        total_score = self.score_sub(self.sub, self.data_directory.joinpath('ground_truth.csv'))
        dict_scores = {} # per tener traccia degli scores
        dict_cluster_dim = {} # per tener traccia della dimensione del cluster
        df_gt = pd.read_csv(self.data_directory.joinpath('ground_truth.csv'))
        gt_dim = df_gt.shape[0]
        for gt in gt_list:
            sub_name = os.path.basename(self.data_directory.joinpath(gt))  # questo serve per estrarre solo in nome, perché per il full se no aggiugne il nome della dir
            sub_name = os.path.splitext(sub_name)[0]
            mrr = self.score_sub(self.sub, self.data_directory.joinpath('gt_clusters', gt), total_score)
            dict_scores[sub_name] = mrr
            tmp = pd.read_csv(self.data_directory.joinpath('gt_clusters', gt))
            cluster_dim = tmp.shape[0]
            dict_cluster_dim[sub_name] = int((cluster_dim*100)/(gt_dim))

        print('\n===========================================')
        print('\033[1;40m RECAP:'+'\033[0;37;40m')
        print(f'\033[1;40m OVERALL SCORE: {total_score}'+'\033[0;37;40m')

        max_score_cluster = max(dict_scores, key=dict_scores.get)
        min_score_cluster = min(dict_scores, key=dict_scores.get)
        x = PrettyTable()
        x.field_names = ['\033[1;40m Cluster'+'\033[0;37;40m', '\033[1;40m Score'+'\033[0;37;40m', '\033[1;40m Dimension'+'\033[0;37;40m']

        for k in dict_scores:
            if k==max_score_cluster:
                x.add_row([f'\033[32;40m {k}'+'\033[0;37;40m', f'\033[32;40m {dict_scores[k]}'+'\033[0;37;40m', f'\033[32;40m {dict_cluster_dim[k]}%'+'\033[0;37;40m'])
            elif k==min_score_cluster:
                x.add_row([f'\033[31;40m {k}'+'\033[0;37;40m', f'\033[31;40m {dict_scores[k]}'+'\033[0;37;40m', f'\033[31;40m {dict_cluster_dim[k]}%'+'\033[0;37;40m'])
            else:
                x.add_row([k, f'{dict_scores[k]}', f'{dict_cluster_dim[k]}%'])
        print(x)
        if save_path is not None:
            with open(f"{save_path}", "w+") as text_file:
                array_strings = [f'\033[33;40m {cool_string}'+'\033[0;37;40m\n', '\n===========================================\n',
                                 '\033[1;40m RECAP:' + '\033[0;37;40m', f'\033[1;40m OVERALL SCORE: {total_score}'+'\033[0;37;40m', x]
                text_file.writelines(array_strings)



    def generate_clusters(self):
        df_test = data.test_df('local')
        df_gt = pd.read_csv(self.data_directory.joinpath('ground_truth.csv'))

        if not os.path.exists(self.data_directory.joinpath('gt_clusters')):
            os.makedirs(self.data_directory.joinpath('gt_clusters'))

        if not os.path.isfile(self.data_directory.joinpath('gt_clusters/gt_only_tablet.csv')):
            print('Creating cluster only on tablet sessions...')
            df_desktop_sessions = df_test[df_test['device']=='tablet']
            sessions = df_desktop_sessions['session_id'].unique().tolist()
            new_gt = df_gt[df_gt['session_id'].isin(sessions)]
            new_gt.to_csv(self.data_directory.joinpath('gt_clusters/gt_only_tablet.csv'), encoding='utf-8', index=False)
            print('Done.')

        if not os.path.isfile(self.data_directory.joinpath('gt_clusters/gt_only_mobile.csv')):
            print('Creating cluster only on mobile sessions...')
            df_desktop_sessions = df_test[df_test['device']=='mobile']
            sessions = df_desktop_sessions['session_id'].unique().tolist()
            new_gt = df_gt[df_gt['session_id'].isin(sessions)]
            new_gt.to_csv(self.data_directory.joinpath('gt_clusters/gt_only_mobile.csv'), encoding='utf-8', index=False)
            print('Done.')

        if not os.path.isfile(self.data_directory.joinpath('gt_clusters/gt_only_desktop.csv')):
            print('Creating cluster only on desktop sessions...')
            df_desktop_sessions = df_test[df_test['device']=='desktop']
            sessions = df_desktop_sessions['session_id'].unique().tolist()
            new_gt = df_gt[df_gt['session_id'].isin(sessions)]
            new_gt.to_csv(self.data_directory.joinpath('gt_clusters/gt_only_desktop.csv'), encoding='utf-8', index=False)
            print('Done.')

        if not os.path.isfile(self.data_directory.joinpath('gt_clusters/gt_only_one_interaction_sessions.csv')):
            #creates ground truth with only one interaction sessions (only the clickout to be predicted)
            print('Creating cluster of one interaction sessions...')
            sessions = df_test.groupby('session_id').size()
            sess = sessions.index[:].tolist()
            number_of_interactions = sessions.iloc[:].tolist()

            d = dict(zip(sess,number_of_interactions))
            one_interaction_sessions = []
            for k in d:
                if d[k]==1:
                    one_interaction_sessions.append(k)
            new_gt = df_gt[df_gt['session_id'].isin(one_interaction_sessions)]
            new_gt.to_csv(self.data_directory.joinpath('gt_clusters/gt_only_one_interaction_sessions.csv'), encoding='utf-8', index=False)
            print('Done.')

        if not os.path.isfile(self.data_directory.joinpath('gt_clusters/gt_only_two_interaction_sessions.csv')):
            #creates ground truth with only two interaction sessions
            print('Creating cluster of two interactions sessions...')
            sessions = df_test.groupby('session_id').size()
            sess = sessions.index[:].tolist()
            number_of_interactions = sessions.iloc[:].tolist()

            d = dict(zip(sess,number_of_interactions))
            one_interaction_sessions = []
            for k in d:
                if d[k]==2:
                    one_interaction_sessions.append(k)
            new_gt = df_gt[df_gt['session_id'].isin(one_interaction_sessions)]
            new_gt.to_csv(self.data_directory.joinpath('gt_clusters/gt_only_two_interaction_sessions.csv'), encoding='utf-8', index=False)
            print('Done.')

        if not os.path.isfile(self.data_directory.joinpath('gt_clusters/gt_less_than_5_interaction_sessions.csv')):
            print('Creating cluster of sessions with less than 5 interactions...')
            sessions = df_test.groupby('session_id').size()
            sess = sessions.index[:].tolist()
            number_of_interactions = sessions.iloc[:].tolist()

            d = dict(zip(sess,number_of_interactions))
            one_interaction_sessions = []
            for k in d:
                if d[k]<5:
                    one_interaction_sessions.append(k)
            new_gt = df_gt[df_gt['session_id'].isin(one_interaction_sessions)]
            new_gt.to_csv(self.data_directory.joinpath('gt_clusters/gt_less_than_5_interaction_sessions.csv'), encoding='utf-8', index=False)
            print('Done.')

        if not os.path.isfile(self.data_directory.joinpath('gt_clusters/gt_5_to_10_interaction_sessions.csv')):
            print('Creating cluster of sessions with number of interactions between 5 and 10...')
            sessions = df_test.groupby('session_id').size()
            sess = sessions.index[:].tolist()
            number_of_interactions = sessions.iloc[:].tolist()

            d = dict(zip(sess,number_of_interactions))
            one_interaction_sessions = []
            for k in d:
                if d[k]<11 and d[k]>4:
                    one_interaction_sessions.append(k)
            new_gt = df_gt[df_gt['session_id'].isin(one_interaction_sessions)]
            new_gt.to_csv(self.data_directory.joinpath('gt_clusters/gt_5_to_10_interaction_sessions.csv'), encoding='utf-8', index=False)
            print('Done.')

        if not os.path.isfile(self.data_directory.joinpath('gt_clusters/gt_more_than_10_interaction_sessions.csv')):
            print('Creating cluster of sessions with less than 10 interactions...')
            sessions = df_test.groupby('session_id').size()
            sess = sessions.index[:].tolist()
            number_of_interactions = sessions.iloc[:].tolist()

            d = dict(zip(sess,number_of_interactions))
            one_interaction_sessions = []
            for k in d:
                if d[k]>10:
                    one_interaction_sessions.append(k)
            new_gt = df_gt[df_gt['session_id'].isin(one_interaction_sessions)]
            new_gt.to_csv(self.data_directory.joinpath('gt_clusters/gt_more_than_10_interaction_sessions.csv'), encoding='utf-8', index=False)
            print('Done.')

        if not os.path.isfile(self.data_directory.joinpath('gt_clusters/ground_truth_small.csv')):
            print('Creating small ground truth...')
            new_gt = df_gt.head(int(df_gt.shape[0]/3))
            new_gt.to_csv(self.data_directory.joinpath('gt_clusters/ground_truth_small.csv'), encoding='utf-8', index=False)
            print('Done.')


        if not os.path.isfile(self.data_directory.joinpath('gt_clusters/gt_no_num_reference_sessions.csv')):
            print('Creating cluster of sessions with no numeric reference...')
            # remove one step sessions
            one_step_count = df_test.groupby('session_id')['step'].count()
            one_step = one_step_count[one_step_count == 1]
            one_step_s = list(one_step.index)
            df = df_test[~df_test['session_id'].isin(one_step_s)]

            # get session with no numeric reference and no one step
            group = df.groupby('session_id')['reference'].apply(
                lambda x: False if x.str.isnumeric().any() else True)
            sess_list = list(group.index)
            index_list = []
            for i in range(len(group)):
                if group.iloc[i] == True:
                    index_list.append(i)
            no_ref_sessions = [sess_list[i] for i in index_list]
            new_gt = df_gt[df_gt['session_id'].isin(no_ref_sessions)]
            new_gt.to_csv(self.data_directory.joinpath('gt_clusters/gt_no_num_reference_sessions.csv'), encoding='utf-8', index=False)

if __name__=='__main__':
    sub_evaluator = SubEvaluator('/home/edoardo/Downloads/2019-06-13_06:30_0.668/lightGBM_prova_06-31-42.csv')
    sub_evaluator.run()
