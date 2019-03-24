from bayes_opt import BayesianOptimization
from functools import partial
from recommenders.collaborative_filterning.itembased import CFItemBased
from utils.writer import Writer


class BayesianValidator:

    def __init__(self, reference_object):
        self.reference_object = reference_object
        self.fixed_params_dict = None
        self.hyperparameters_dict = None
        self.writer = Writer(file_base_path='validation_result',
                             file_name=str(reference_object.__class__).split('.')[-1].replace('\'>', ''))

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
        self.writer.write_line('params: {}\n MRR is: {}\n\n'.format(params_dict, score))
        del model
        return score







