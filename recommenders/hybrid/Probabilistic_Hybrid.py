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

    def __init__(self, mode='local'):
        name = 'probabilistic_hybrid'
        cluster = 'no_cluster'
        super(Probabilistic_Hybrid, self).__init__(mode, cluster, name)

        self.current_directory = Path(__file__).absolute().parent
        self.data_directory = self.current_directory.joinpath('..', '..', 'submissions/hybrid')

        self.mode = mode
        self.dfs_subname = []

        """Set key as the name of the submission file w/o '.csv'; its value corresponds to the weight assigned to that submission.
        If no key-value pair is set, then the weight = 1 by default"""

        params = {
            'content_based_old': 0.0032,
            'last_interaction': 0.7,
            'lazyUserRec': 1,
            'location_subm': 0.9,
            'min_price_based': 1.5,
            'xgboostlocal': 2
        }

        num_file = 0
        directory = self.data_directory.joinpath(self.mode)
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv"):
                    tmp = pd.read_csv(directory.joinpath(file))
                    tmp = tmp.sort_values(by=['user_id', 'timestamp'])
                    tmp = tmp.reset_index(drop=True)
                    sub_name = os.path.basename(directory.joinpath(file))  # questo serve per estrarre solo in nome, perché per il full se no aggiugne il nome della dir
                    sub_name = os.path.splitext(sub_name)[0]  # questo rimuove il .csv dal nome
                    if (num_file <= len(params)):
                        self.dfs_subname.append([tmp, sub_name, params[sub_name]])
                    else:
                        self.dfs_subname.append([tmp, sub_name, 1])
                    num_file += 1

    def fit(self):
        gt_csv = self.data_directory.joinpath('ground_truth.csv')
        data_directory = self.data_directory.joinpath(self.mode)
        self.dict_sub_scores = {}
        for d in self.dfs_subname:
            print(f'Getting scores for {d[1]}...')
            if os.path.isfile(self.data_directory.joinpath(f'scores/scores_{d[1]}.pkl')):
                with open(self.data_directory.joinpath(f'scores/scores_{d[1]}.pkl'), 'rb') as file:
                    self.dict_sub_scores[d[1]] = pickle.load(file)
            else:
                scores = []
                # TODO add get result for first ref  --> Matte More
                for n in tqdm(range(1, 25)):
                    subm_csv = self.data_directory.joinpath(f'scores/item_{d[1]}_{n}.csv')
                    mrr = f.score_submissions(subm_csv, gt_csv, f.get_reciprocal_ranks)
                    scores.append(mrr)
                self.dict_sub_scores[d[1]] = scores
                print('Saving list...')
                with open(self.data_directory.joinpath(f'scores/scores_{d[1]}.pkl'), 'wb') as file:
                    pickle.dump(scores, file)
                # Remove files scores/item_{}
                for n in tqdm(range(1, 25)):
                    os.remove(self.data_directory.joinpath(f'scores/item_{d[1]}_{n}.csv'))
        return self.dict_sub_scores


    def recommend_batch(self):
        exp = 0.5  # esponente della radice
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
                tmp_list = [int(x) for x in tmp_list]
                tmp_dict = dict(zip(tmp_list, self.dict_sub_scores[key][:len(tmp_list)]))
                impressions_scores_dicts.append(tmp_dict)

            final_dict = Counter({})  # devo inizializzarlo per sommarlo
            for j in impressions_scores_dicts:
                final_dict += Counter(j)
            final_dict = dict(final_dict)

            final_dict = sorted(final_dict.items(), key=lambda x: float(x[1]))

            impressions_reordered = [i[0] for i in final_dict]
            impressions_reordered = impressions_reordered[::-1]
            string_impressions_reordered = " ".join(str(x) for x in impressions_reordered)
            recommendation_list.append(string_impressions_reordered)

        item_rec['item_recommendations'] = recommendation_list

        submission = pd.concat([submission, item_rec], axis=1)

        if self.mode == 'local':
            print('Computing the score...')
            self.score_sub(submission)
        elif self.mode == 'full':
            print('Computing the full score...')

        print('DONE.')
        return submission


    def score_sub(self, submission):
        gt_csv = self.data_directory.joinpath('ground_truth.csv')
        mrr = f.score_submissions(submission, gt_csv, f.get_reciprocal_ranks, subm_csv_is_file=False)
        print(f'Score: {mrr}')

    def get_scores_batch(self):
        return


if __name__=='__main__':

    model = Probabilistic_Hybrid(mode='local')
    model.run()
