import data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from tqdm.auto import tqdm

def hybrid(scores_array):
    label = pd.read_csv('label_full.csv')
    impression_position = pd.read_csv('impression_position.csv')

    def compute_mrr(df):
        rr = 1 / df[df['label'] == 1]['ranking_pos'].values
        print(np.sum(rr) / len(rr))

    def normalize_scores(df):
        cols = [c for c in df.columns if 'score' in c]

        scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
        scaler.fit(df[cols])
        df[cols] = scaler.transform(df[cols]).flatten()
        print(df[cols])

        # scaler = StandardScaler()
        # scaler.fit(df[cols])
        # df[cols] = scaler.transform(df[cols]).flatten()
        # print(df[cols])

        return df

    def add_ranking_position(df, score_cols=None):
        print(df.columns)
        if score_cols is None:
            score_cols = [c for c in df.columns if 'score' in c]
        else:
            score_cols = [score_cols]
        df = df.sort_values(['user_id', 'session_id', score_cols[0]], ascending=False)

        actual_u = 'sorpresina'
        actual_s = 'baby'
        position = []
        score_nobias = []
        count = 1
        for (u, s) in tqdm(zip(df['user_id'], df['session_id'])):
            if u != actual_u or s != actual_s:
                actual_u = u
                actual_s = s
                count = 1
            position.append(count)
            count += 1
        df['ranking_pos'] = position
        return df

    def compute_confidence(df):
        recall_list = []
        precision_list = []
        clicked = df[df['label'] == 1]
        for i in tqdm(range(25)):
            tot_pred = len(df[df['ranking_pos'] == 1 & (df['impression_position'] == i + 1)])
            num_prec = len(df[df['ranking_pos'] == 1 & (df['impression_position'] == i + 1) & (df['label'] == 1)])

            tot = len(clicked[clicked['impression_position'] == i + 1])
            predicted = len(clicked[(clicked['impression_position'] == i + 1) & (clicked['ranking_pos'] == 1)])

            if tot == 0:
                tot = 1
            if tot_pred == 0:
                tot_pred = 1

            recall_list.append(predicted / tot)
            precision_list.append(num_prec / tot_pred)
        recall_list = np.array(recall_list)
        precision_list = np.array(precision_list)
        # confidence = (np.multiply(precision_list,recall_list)*2)/np.sum([recall_list, precision_list])
        print(f'precision:{precision_list}')
        print(f'recall:{recall_list}')

        # set the confidence equal to the precision
        confidence = precision_list
        return confidence

    def compute_confidence_score(df, confidence):
        score_cols = [c for c in df.columns if 'score' in c]
        confidence_score = []
        count = None
        for (s, rank, impr_pos) in tqdm(zip(df[score_cols[0]], df['ranking_pos'], df['impression_position'])):
            if rank == 1:
                count = impr_pos - 1
            confidence_score.append(s * confidence[count])
        df[score_cols] = confidence_score
        return df

    scores_prep_list = []

    for score in scores_array:
        # normalize the scores_array
        score = score.merge(label)
        score = score.merge(impression_position)

        score = normalize_scores(score)

        print(score)
        # add ranking pos
        score = add_ranking_position(score)

        # compute confidence
        confidence = compute_confidence(score)

        # compute confidence score and append the preprocessed score on the list
        score = compute_confidence_score(score, confidence)

        # print(score)
        scores_prep_list.append(score)

    final_score = scores_prep_list[0]

    for score in scores_prep_list[1:]:
        final_score = pd.merge(final_score, score, on=['user_id', 'session_id', 'item_id'])

    cols = [c for c in final_score.columns if 'score' in c]

    f_score = None
    for c in cols:
        if f_score is None:
            f_score = final_score[c]
        else:
            f_score += final_score[c]

    final_score['final_score'] = f_score
    final_score.rename(columns={'label_y': 'label'}, inplace=True)

    final_score = add_ranking_position(final_score, 'final_score')
    compute_mrr(final_score)
    return final_score


if __name__ == '__main__':
    #scores_cat = pd.read_csv('catboost_rank.csv.gz')  # , nrows=100000)
    # scores_rnn = pd.read_csv('rnn_GRU_2layers_64units_2dense_noclass0.csv.gz', nrows=100000)
    scores_xg = pd.read_csv('xgb_forte_700.csv.gz')  # , nrows=100000)
    #scores_tf = pd.read_csv('scores_pairwise.csv.gz')
    scores_tf2 = pd.read_csv('scores_softmax.csv.gz')
    a = hybrid([scores_tf2, scores_xg])