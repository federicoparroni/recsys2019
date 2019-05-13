import utils.functions as f
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from prettytable import PrettyTable
import pickle

# stringa fatta su http://patorjk.com/software/taag/#p=display&f=Standard&t=Sub%20Evaluator%0A (font Standard)

cool_string = """
  ____           _ _   _               _____            _             _
 |  _ \ ___  ___(_) |_(_) ___  _ __   | ____|_   ____ _| |_   _  __ _| |_ ___  _ __
 | |_) / _ \/ __| | __| |/ _ \| '_ \  |  _| \ \ / / _` | | | | |/ _` | __/ _ \| '__|
 |  __/ (_) \__ \ | |_| | (_) | | | | | |___ \ V / (_| | | |_| | (_| | || (_) | |
 |_|   \___/|___/_|\__|_|\___/|_| |_| |_____| \_/ \__,_|_|\__,_|\__,_|\__\___/|_|

"""
class ScorePerPositionEvaluator():
    """
    Evaluator of a local submission.
    You need the scores folder:
        submissions
            |__hybrid
                    |__scores
    You need an evaluator folder in submissions:

        submissions
            |__evaluator

    In the evaluator folder, you need:
    - the submission
    - the test set
    - the ground truth file (test with the missing clickouts)
    """

    def __init__(self, sub_name):
        self.current_directory = Path(__file__).absolute().parent
        self.data_directory = self.current_directory.joinpath('..', 'submissions/hybrid/scores')
        self.sub_name = sub_name
        self.gt_csv = self.current_directory.joinpath('..', 'submissions/evaluator', 'ground_truth.csv')
        self.sub_scores = []
        self.subm_csv = self.current_directory.joinpath('..', 'submissions/evaluator', f'{sub_name}.csv')
        self.overall_score = f.score_submissions(self.subm_csv, self.gt_csv, f.get_reciprocal_ranks)

    def run(self):
        print(f'\033[1;34;40m {cool_string}'+'\033[0;37;40m')
        print(f'Global sub localc score: {self.overall_score}')
        self.get_scores()

        x = PrettyTable()
        x.field_names = ['\033[1;40m Position'+'\033[0;37;40m', '\033[1;40m Score'+'\033[0;37;40m', '\033[1;40m Perc. Overall'+'\033[0;37;40m']

        for i in range(len(self.sub_scores)):
            x.add_row([i+1, f'{self.sub_scores[i]}', '{:.2f}%'.format(self.sub_scores[i]*100/self.overall_score)])
        print(x)
        self.plot()

    def plot(self):
        # Draw plot
        import matplotlib.pyplot as plt
        df = pd.DataFrame(columns=['Position', 'Score'])
        for i in range(len(self.sub_scores)):
            df = df.append({'Position' : i+1, 'Score' : self.sub_scores[i]} , ignore_index=True)
        fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
        ax.vlines(x=df.index, ymin=0, ymax=df.Score, color='firebrick', alpha=0.7, linewidth=2)
        ax.scatter(x=df.index, y=df.Score, s=75, color='firebrick', alpha=0.7)

        # Title, Label, Ticks and Ylim
        ax.set_title(f'Position for {self.sub_name} ({round(self.overall_score, 3)})', fontdict={'size':22})
        ax.set_ylabel('Score')
        ax.set_xticks(df.index)
        pos = [24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]
        pos = sorted(pos)
        pos = [str(p) for p in pos]
        ax.set_xticklabels(pos, rotation=60, fontdict={'horizontalalignment': 'right', 'size':12})
        ax.set_ylim(0, max(self.sub_scores)+0.05)

        # Annotate
        for row in df.itertuples():
            ax.text(row.Index, row.Score, s=round(row.Score, 3), horizontalalignment= 'center', verticalalignment='bottom', fontsize=12)

        plt.show()

    def get_scores(self):
        print(f'Getting scores for {self.sub_name}...')
        if os.path.isfile(self.data_directory.joinpath(f'scores_{self.sub_name}.pkl')): # if the scores were previously computed, simply loads them
            with open(self.data_directory.joinpath(f'scores_{self.sub_name}.pkl'), 'rb') as file:
                self.sub_scores = pickle.load(file)
        else:
            # if there are no scores previously computed...
            # first: generate a sub for each impression "column"
            # second: score each sub
            # third: append each score to the scores' list
            scores = []
            self.generate_column_subs()
            for n in tqdm(range(1, 25)):
                subm_csv = self.data_directory.joinpath(f'item_{self.sub_name}_{n}.csv')
                mrr = f.score_submissions(subm_csv, self.gt_csv, f.get_reciprocal_ranks)
                scores.append(mrr)
            self.sub_scores = scores
            print('Saving list...')
            with open(self.data_directory.joinpath(f'scores_{self.sub_name}.pkl'), 'wb') as file:
                pickle.dump(scores, file)
            for n in tqdm(range(1, 25)):
                os.remove(self.data_directory.joinpath(f'item_{self.sub_name}_{n}.csv'))

    def generate_column_subs(self):
        #takes a list as: dataframe, submission name, coefficient
        #creates 25 subs, 1 for each "column"
        sub = pd.read_csv(self.current_directory.joinpath('..', 'submissions/evaluator', f'{self.sub_name}.csv')) # the dataframe
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
            new_sub.to_csv(self.data_directory.joinpath(f'item_{self.sub_name}_{n}.csv'), encoding='utf-8', index=False)

if __name__=='__main__':
    eval = ScorePerPositionEvaluator('xgboostlocal')
    eval.run()
