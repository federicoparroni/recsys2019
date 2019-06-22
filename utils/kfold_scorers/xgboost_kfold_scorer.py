from recommenders.kfold_scorer import KFoldScorer
from recommenders.XGBoost import XGBoostWrapper
from utils.dataset import DatasetXGBoost
from utils.menu import mode_selection
from utils.menu import cluster_selection
from utils.menu import single_choice

if __name__ == "__main__":
    kind = input('insert the kind: ')
    mode = mode_selection()
    cluster = cluster_selection()
    dataset = DatasetXGBoost(mode, cluster, kind)

    init_params = {
        'mode': mode, 
        'cluster': cluster, 
        'kind': kind, 
        'ask_to_load': False, 
        'class_weights': False,
        #'learning_rate': 0.3, 
        #'min_child_weight': 1, 
        #'n_estimators': 10,
        #'max_depth': 3, 
        #'subsample': 1, 
        #'colsample_bytree': 1, 
        #'reg_lambda': 1, 
        #'reg_alpha': 0,
    }

    fit_params = {}

    kfscorer = KFoldScorer(model_class=XGBoostWrapper, init_params=init_params, k=5)
    kfscorer.fit_predict(dataset, fit_params=fit_params)
