from recommenders.lightGBM import lightGBM
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import data
from utils.menu import single_choice

class OptimizerWrapper:
    def __init__(self, recommender_class, mode, cluster):
        self.space, self.objective = recommender_class.get_optimize_params(mode, cluster)
    def optimize(self):
        best_param = gp_minimize(self.objective, self.space, n_random_starts=10, n_calls=100)
        print(best_param)


if __name__ == '__main__':
    mode=single_choice('insert mode:', ['local', 'small'])
    cluster = single_choice('insert cluster', ['no_cluster'])
    opt = OptimizerWrapper(lightGBM, mode=mode, cluster=cluster)
    opt.optimize()
