from recommenders.lightGBM import lightGBM
from skopt import gp_minimize
from skopt import dummy_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import data
from utils.menu import single_choice

class OptimizerWrapper:
    def __init__(self, recommender_class, mode, cluster, dataset_name):
        self.space, self.objective = recommender_class.get_optimize_params(mode, cluster, dataset_name)
    def optimize_bayesian(self):
        best_param = gp_minimize(self.objective, self.space, n_random_starts=10, n_calls=100)
        print(best_param)
    def optimize_random(self):
        best_param = dummy_minimize(self.objective, self.space, n_calls=1000)


if __name__ == '__main__':
    opt_technique = single_choice('optimization technique', ['bayesian', 'random'])
    mode = single_choice('insert mode:', ['local', 'small'])
    cluster = single_choice('insert cluster', ['no_cluster'])
    dataset_name = input('insert_the_dataset_name')
    opt = OptimizerWrapper(lightGBM, mode=mode, cluster=cluster, dataset_name=dataset_name)
    if opt_technique == 'bayesian':
        opt.optimize_bayesian()
    else:
        opt.optimize_random()

