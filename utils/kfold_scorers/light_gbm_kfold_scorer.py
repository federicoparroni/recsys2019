from recommenders.kfold_scorer import KFoldScorer
from recommenders.lightGBM import lightGBM
from utils.dataset import DatasetLightGBM
from utils.menu import mode_selection
from utils.menu import cluster_selection
from utils.menu import single_choice

if __name__ == "__main__":
    mode = mode_selection()
    cluster = cluster_selection()
    dataset_name = input('insert the dataset name:\n')
    dataset = DatasetLightGBM(mode, cluster, dataset_name)

    params_dict = {
        'boosting_type':'gbdt',
        'num_leaves': 21,
        'max_depth': -1,
        'learning_rate': 0.1,
        'n_estimators': 774,
        'subsample_for_bin': 200000,
        'class_weights': None,
        'min_split_gain': 0.0,
        'min_child_weight': 0.0,
        'min_child_samples': 20,
        'subsample':1.0,
        'subsample_freq': 0,
        'colsample_bytree': 1,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'random_state': None,
        'n_jobs': -1,
        'silent': False,
        'importance_type': 'split',
        'metric': 'None',
        'print_every': 1000,
        'first_only': True
    }

    init_params = {
        'mode': mode, 
        'cluster': cluster, 
        'dataset_name': dataset_name,
        'params_dict': params_dict,
    }

    fit_params = {}

    kfscorer = KFoldScorer(model_class=lightGBM, init_params=init_params, k=5)
    kfscorer.fit_predict(dataset, fit_params=fit_params)
