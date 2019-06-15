from recommenders.kfold_scorer import KFoldScorer
from recommenders.XGBoost_Classifier import XGBoostWrapperClassifier
from utils.dataset import DatasetXGBoost, DatasetXGBoostClassifier
from utils.menu import mode_selection
from utils.menu import cluster_selection
from utils.menu import single_choice

if __name__ == "__main__":
    mode = mode_selection()
    cluster = cluster_selection()
    dataset = DatasetXGBoostClassifier(mode, cluster)

    init_params = {
        'mode': mode,
        'cluster': cluster,
        #'ask_to_load': False,
        #'class_weights': False,
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

    kfscorer = KFoldScorer(model_class=XGBoostWrapperClassifier, init_params=init_params, k=5)
    kfscorer.fit_predict(dataset, fit_params=fit_params)
