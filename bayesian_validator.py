from recommenders.XGBoost import XGBoostWrapper
from bayes_opt import BayesianOptimization
from functools import partial
from utils.writer import writer
import gc
import utils.telegram_bot as HERA


class BayesianValidator:

    def __init__(self, reference_object):
        self.reference_object = reference_object
        self.fixed_params_dict = None
        self.hyperparameters_dict = None

        # used to know where to store the validation result
        self.file_base_path = 'validation_result'
        self.file_name = str(reference_object.__class__).split(
            '.')[-1].replace('\'>', '')
        self.writer = writer(
            file_base_path=self.file_base_path, file_name=self.file_name)

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
        gc.collect()
        self.writer.write(
            'params: {}\n MRR is: {}\n\n'.format(params_dict, score))

        # sending a message on the telegram channel
        HERA.send_message(
            'params: {}\n MRR is: {}\n\n'.format(params_dict, score))

        return score


if __name__ == "__main__":
    m = XGBoostWrapper(mode='small')
    v = BayesianValidator(m)
    v.validate(100)
