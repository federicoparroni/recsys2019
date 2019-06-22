import data
import pandas as pd
from tqdm import tqdm
import numpy as np
from extract_features.label import ImpressionLabel

def _compute_rr(label, score):
    label = np.array(label)
    score = np.array(score)
    our_guess = score[label == 1]
    t = 0
    for s in score:
        if s >= our_guess:
            t += 1
    if t >= 1:
        return 1/t
    else:
        return 0

def _infer_score_column_name(df):
    return df.columns.values[-1]

def _compute_mrr(a):
    a_user = a.head(1).user_id.values[0]
    a_sess = a.head(1).session_id.values[0]
    rrs = []
    label = []
    score = []
    for r in zip(a.user_id, a.session_id, a.label, a[_infer_score_column_name(a)]):
        if r[0] == a_user and r[1] == a_sess:
            label.append(int(r[2]))
            score.append(float(r[3]))
        else:
            rr = _compute_rr(label, score)
            if rr > 0:
                rrs.append(rr)
            label = [r[2]]
            score = [r[3]]
            a_user = r[0]
            a_sess = r[1]
    mrr = sum(rrs)/len(rrs)
    return mrr

def get_scores_mrr(mode, scores):
    o = ImpressionLabel(mode)
    f = o.read_feature()
    if isinstance(scores, str):
        scores = pd.read_csv(scores)
    m = f.merge(scores, how='left')
    m = m.dropna()
    m = m.groupby(['user_id', 'session_id', 'item_id']).last().reset_index()
    tl = pd.read_csv(f'dataset/preprocessed/no_cluster/{mode}/test.csv', usecols=['user_id', 'session_id'])
    
    print('mrr in full train data is: {}'.format(_compute_mrr(m)))

    a = m[m.user_id.isin(tl.user_id.values) & m.session_id.isin(tl.session_id.values)]
    print('mrr in just local test (ie a slice of full train) is: {}'.format(_compute_mrr(a)))


if __name__ == "__main__":
    path = input('put the path of the score feature to eval: ')

    get_scores_mrr(mode='local', score_path=path)
