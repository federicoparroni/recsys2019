import data
import time
import pandas as pd
from tqdm.notebook import tqdm
from recommenders.XGBoost import XGBoostWrapper
from eli5.sklearn.permutation_importance import PermutationImportance
import pickle

model = XGBoostWrapper('local')
model.xg.load_model('models/final_stacking.model')

# load dataset
X_test, y_test, _, _ = data.dataset_xgboost_test(mode='local', kind='all')

X_test_dense = X_test.todense()

i = 0
def score_fn(model, X, y):
    global i
    print(i)
    i += 1
    t0 = time.time()
    
    target_indices = data.target_indices('local')
    full_impressions = data.full_df()
    
    scores = list(model.xg.predict(X))
    
    final_predictions = []
    count = 0
    for index in tqdm(target_indices):
        impressions = list(
            map(int, full_impressions.loc[index]['impressions'].split('|')))
        predictions = scores[count:count + len(impressions)]
        couples = list(zip(predictions, impressions))
        couples.sort(key=lambda x: x[0], reverse=True)
        _, sorted_impr = zip(*couples)
        final_predictions.append((index, list(sorted_impr)))
        count = count+len(impressions)
    
    mrr = model.compute_MRR(final_predictions)
    print('Done in', time.time() - t0)
    print()
    
    return mrr

perm_importance = PermutationImportance(model, score_fn)
perm_importance.fit(X_test_dense, y_test)

with open('feat_imp.model', 'wb') as file:
    pickle.dump(perm_importance, file)






# df = pd.read_hdf('dataset/preprocessed/no_cluster/local/xgboost/base_dataset_stacking/base.hdf', 'train')