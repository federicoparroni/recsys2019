import pandas as pd
import data
from tqdm import tqdm
from sklearn.metrics import classification_report

"""
    given a submission on local or small, prints out the precision and recall in case the 
    clicked impression is the first of the list, the second, ecc ..
"""
def classification_report_on_impr_position(sub_path):
    train = data.train_df(mode='full')
    sub = pd.read_csv(sub_path)
    merged = train.merge(sub).drop(['timestamp', 'step', 'action_type', 'platform', 'city', 'device', 'current_filters', 'prices', 'frequence'], axis=1)

    y_true = []
    y_pred = []
    for idx, row in tqdm(merged.iterrows()):
        ref = row.reference
        imp = row.impressions.split('|')
        rec = row.item_recommendations.split(' ')
        y_true.append(imp.index(ref)+1)
        y_pred.append(imp.index(rec[0]) + 1)

    labels = ['index of clicked impression: {}'.format(x+1) for x in list(range(25))]
    s = classification_report(y_true, y_pred, target_names=labels)
    print(s)

if __name__=='__main__':
    classification_report_on_impr_position('/home/giovanni/Downloads/xgb14f0666local.csv')