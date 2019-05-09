from recommenders.recommender_base import RecommenderBase
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import Counter
import pickle
import os
import utils.functions as f

class Probabilistic_Hybrid(RecommenderBase):
    """Merge scores of submissions located in submissions/hybrid/(local or full): putting there submissions you want to merge"""

    """TODO Creare una sottocartella di submissions chiamata hybrid: creare altre due sottocartelle di hybrid chiamate local e full
            submissions
                |__hybrid
                        |__full
                        |__local
                        |__scores

    Questa classe utilizza il file ground_truth.csv per calcolare lo score locale, NON utilizza il metodo evaluate della classe base: la
    chiamata run() con la mode='local' calcola lo score in locale."""

    def __init__(self, params, mode='local'):
        name = 'probabilistic_hybrid'
        cluster = 'no_cluster'
        super(Probabilistic_Hybrid, self).__init__(mode, cluster, name)

        self.current_directory = Path(__file__).absolute().parent
        self.data_directory = self.current_directory.joinpath('..', '..', 'submissions/hybrid')

        self.mode = mode
        self.dfs_subname = []

        """Set key as the name of the submission file w/o '.csv'; its value corresponds to the weight assigned to that submission.
        If no key-value pair is set, then the weight = 1 by default"""

        #TODO find the optimal parameters with the Bayesian
        """
        params = {
            'content_based_old': 0.0032,
            'last_interaction': 0.7,
            'lazyUserRec': 1,
            'location_subm': 1,
            'min_price_based': 1.5,
            'xgboostlocal': 1.8,
            'RNN_local': 0
        }


        params = {
            'content_based_old': 0,
            'last_interaction': 1,
            'lazyUserRec': 1,
            'location_subm': 0,
            'min_price_based': 0,
            'xgboostlocal': 0,
            'RNN_local': 0.5
        }
        """
        self.params = params
        common_sessions = self.check_submissions()

        num_file = 0
        directory = self.data_directory.joinpath(self.mode)
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv"):
                    # This for creates a list of lists, "dfs_subname"
                    # each list element contains
                    # - the dataframe of the submission
                    # - the name of the submission
                    # - the coefficient used for the submission weight
                    tmp = pd.read_csv(directory.joinpath(file))
                    tmp = tmp[tmp['session_id'].isin(common_sessions)]
                    tmp = tmp.sort_values(by=['user_id', 'timestamp'])
                    tmp = tmp.reset_index(drop=True)
                    sub_name = os.path.basename(directory.joinpath(file))  # questo serve per estrarre solo in nome, perché per il full se no aggiugne il nome della dir
                    sub_name = os.path.splitext(sub_name)[0]  # questo rimuove il .csv dal nome
                    if (num_file <= len(params)):
                        self.dfs_subname.append([tmp, sub_name, params[sub_name]])
                    else:
                        self.dfs_subname.append([tmp, sub_name, 1]) # in case the param is not specified in the dictonary for that submission
                    num_file += 1

    def fit(self):
        # Note: the 'ground_truth.csv' file is the test WITH the references that have been removed
        # Ask to @teomore to have it
        gt_csv = self.data_directory.joinpath('ground_truth.csv')
        data_directory = self.data_directory.joinpath(self.mode)
        self.dict_sub_scores = {}
        # This creates a dictonary that holds for each sub the list of the scores (1 for each impression)
        for d in self.dfs_subname:
            print(f'Getting scores for {d[1]}...')
            if os.path.isfile(self.data_directory.joinpath(f'scores/scores_{d[1]}.pkl')): # if the scores were previously computed, simply loads them
                with open(self.data_directory.joinpath(f'scores/scores_{d[1]}.pkl'), 'rb') as file:
                    self.dict_sub_scores[d[1]] = pickle.load(file)
            else:
                # if there are no scores previously computed...
                # first: generate a sub for each impression "column"
                # second: score each sub
                # third: append each score to the scores' list
                scores = []
                self.generate_column_subs(d)
                for n in tqdm(range(1, 25)):
                    subm_csv = self.data_directory.joinpath(f'scores/item_{d[1]}_{n}.csv')
                    mrr = f.score_submissions(subm_csv, gt_csv, f.get_reciprocal_ranks)
                    scores.append(mrr)
                self.dict_sub_scores[d[1]] = scores
                print('Saving list...')
                with open(self.data_directory.joinpath(f'scores/scores_{d[1]}.pkl'), 'wb') as file:
                    pickle.dump(scores, file)
                # Remove files scores/item_{d[1]}_{n}
                for n in tqdm(range(1, 25)):
                    os.remove(self.data_directory.joinpath(f'scores/item_{d[1]}_{n}.csv'))
        return self.dict_sub_scores

    def check_submissions(self):
        print('Checking submissions...')
        gt_csv = self.data_directory.joinpath('ground_truth.csv')
        df_gt= pd.read_csv(gt_csv)
        sessions = df_gt['session_id'].unique().tolist()
        for root, dirs, files in os.walk(self.data_directory.joinpath(self.mode)):
            for file in files:
                if file.endswith(".csv"):
                    tmp = pd.read_csv(self.data_directory.joinpath(self.mode, file))
                    tmp_sessions = tmp['session_id'].unique().tolist()
                    common_sessions = set(sessions) & set(tmp_sessions)
                    sessions = list(common_sessions)
        return sessions

    def run_hybrid(self):
        self.fit()
        sub = self.recommend_batch()
        MRR = self.score_sub(sub)

    def recommend_batch(self):
        exp = 0.5  # we'll compute the squared radix
        for d in self.dfs_subname:
            scores = self.dict_sub_scores[d[1]]  #  d[1] è il nome della sub
            scores = [float(x) * d[2] for x in scores]  # d[2] è lo score
            scores = [float((e) ** (exp)) for e in
                      scores]  # ci faccio la radice quadrata ma magari qualcos'altro funziona meglio
            self.dict_sub_scores[d[1]] = scores

        submission = self.dfs_subname[0][0][['user_id', 'session_id', 'timestamp',
                                             'step']]  # la sub ibrida avrà le stesse colonne delle sub tranne item_recommendations

        item_recommendations = {}  # dizionario del tipo 'nome_sub': [colonna raccomandazioni]
        print('Getting the recommendations...')
        for d in self.dfs_subname:
            item_recommendations[d[1]] = d[0]['item_recommendations']
        columns = ['item_recommendations']
        item_rec = pd.DataFrame(columns=columns)

        recommendation_list = []
        print('Reordering the impression list...')
        for i in tqdm(range(len(submission['session_id']))):
            impressions_scores_dicts = []  # list of dictonaries 'impression':'score'
            for key in item_recommendations:
                tmp_list = str(item_recommendations[key][i]).split()
                tmp_list = [int(x) for x in tmp_list if x!='nan']
                tmp_dict = dict(zip(tmp_list, self.dict_sub_scores[key][:len(tmp_list)]))
                impressions_scores_dicts.append(tmp_dict)

            # each dictonary holds the scores for a specific sub, we will sum all the scores
            # to get the final reordering of the impressions
            final_dict = Counter({})
            for j in impressions_scores_dicts:
                final_dict += Counter(j)
            final_dict = dict(final_dict)

            final_dict = sorted(final_dict.items(), key=lambda x: float(x[1]))
            # get the impressions' list in descending order
            impressions_reordered = [i[0] for i in final_dict]
            impressions_reordered = impressions_reordered[::-1]
            string_impressions_reordered = " ".join(str(x) for x in impressions_reordered)
            recommendation_list.append(string_impressions_reordered)

        item_rec['item_recommendations'] = recommendation_list

        submission = pd.concat([submission, item_rec], axis=1)

        # TODO: DECOMMENTA PER PROVARE DA SOLO SENZA OPTIMIZER
        #if self.mode == 'local':
        #    print('Computing the score...')
        #    mrr =self.score_sub(submission)

        print('DONE.')
        return submission


    def score_sub(self, submission):
        #compute the score of a submission using utils/functions.py
        gt_csv = self.data_directory.joinpath('ground_truth.csv')
        mrr = f.score_submissions(submission, gt_csv, f.get_reciprocal_ranks, subm_csv_is_file=False)
        print(f'Score: {mrr}')
        return mrr

    def generate_column_subs(self, d):
        #takes a list as: dataframe, submission name, coefficient
        #creates 25 subs, 1 for each "column"
        sub = d[0] # the dataframe
        item_rec = sub['item_recommendations']
        for n in range(1,25):
            rec_list = []
            for i in item_rec:
                l = str(i).split()
                if len(l)>n:
                    e = l[n-1]
                else:
                    e = 'a' # questo mi serve perché così tolgo le righe con le 'a'
                rec_list.append(e)

            new_sub = sub[['user_id','session_id', 'timestamp','step']]
            new_sub['item_recommendations'] = rec_list
            new_sub = new_sub[new_sub['item_recommendations'] != 'a'] # in questo modo valuto solo sulle righe che hanno n impressions
            new_sub.to_csv(self.data_directory.joinpath(f'scores/item_{d[1]}_{n}.csv'), encoding='utf-8', index=False)

    def get_scores_batch(self):
        return


if __name__=='__main__':
    params = {
        'content_based_old': 0,
        'last_interaction': 1,
        'lazyUserRec': 1,
        'location_subm': 1,
        'min_price_based': 1,
        'xgboostlocal': 1,
        'RNN_local': 0,
        'tf_loc': 1
    }
    model = Probabilistic_Hybrid(params, mode='local')
    model.run_hybrid()
