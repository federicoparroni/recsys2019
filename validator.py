from bayes_opt import BayesianOptimization
from functools import partial
from recommenders.collaborative_filterning.itembased import CFItemBased


class BayesianValidator:

    def __init__(self, reference_object):
        self.reference_object = reference_object
        self.fixed_params_dict = None
        self.hyperparameters_dict = None

    def validate(self, iterations):
        self.fixed_params_dict, self.hyperparameters_dict = self.reference_object.get_params()
        optimizer = BayesianOptimization(
            f=self._validate_step,
            pbounds=self.hyperparameters_dict,
            random_state=1,
        )
        optimizer.maximize(
            init_points=2,
            n_iter=iterations,
        )

        print(optimizer.max)
        return optimizer

    def _validate_step(self, **dict):
        # initialize the recommender
        params_dict = {**self.fixed_params_dict, **dict}
        #partial_initialized_model = partial(self.reference_object.__init__, **self.fixed_params_dict)
        model = self.reference_object.__class__(**params_dict)
        score = model.evaluate()
        del model
        return score







