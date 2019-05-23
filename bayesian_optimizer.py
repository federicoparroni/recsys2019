from recommenders.lightGBM import lightGBM
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import data


class OptimizerWrapper:
    def __init__(self, recommender_class):
        self.space, self.objective = recommender_class.get_optimize_params()
        a=4
    def optimize(self):
        best_param = gp_minimize(self.objective, self.space, n_random_starts=10, n_calls=10)
        print(best_param)


if __name__ == '__main__':
    opt = OptimizerWrapper(lightGBM)
    opt.optimize()
