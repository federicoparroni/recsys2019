from recommenders.XGBoost import XGBoostWrapper
from functools import partial

# from recommenders.catboost_rank import CatboostRanker
from utils.writer import writer
import gc
import utils.telegram_bot as HERA
from random import randint
from utils.automatic_sub_exporter import AutomaticSubExporter


class RandomValidator:

    """
    searches the parameter space randomly.
    params can be specified directly in the recommender class that I want to validate 
    (check out one that has those already specified)
    """

    def __init__(self, reference_object, granularity=100, automatic_export=True):
        self.reference_object = reference_object
        self.granularity = granularity

        # used to know where to store the validation result
        self.file_base_path = 'validation_result'
        self.file_name = str(reference_object.__class__).split(
            '.')[-1].replace('\'>', '')
        self.writer = writer(
            file_base_path=self.file_base_path, file_name=self.file_name)

        self.automatic_export = None
        if automatic_export:
            self.automatic_export = AutomaticSubExporter(reference_object)

    def sample(self, tuple):
        low_bound = tuple[0]
        upp_bound = tuple[1]
        rand_numb = randint(0, self.granularity)
        step = (upp_bound - low_bound)/self.granularity
        return low_bound + step*rand_numb

    def validate(self, iterations):
        self.fixed_params_dict, self.hyperparameters_dict = self.reference_object.get_params()
        for i in range(iterations):
            # sample a random parameter from the dictionary
            sampled_params = {}
            for key, value in self.hyperparameters_dict.items():
                sampled_params[key] = self.sample(value)

            params_dict = {**self.fixed_params_dict, **sampled_params}
            print(params_dict)
            model = self.reference_object.__class__(**params_dict)
            score = model.evaluate()

            if self.automatic_export != None:
                self.automatic_export.check_if_export(score, params_dict)
            
            del model
            gc.collect()
            self.writer.write(
                'params: {}\n MRR is: {}\n\n'.format(params_dict, score))

            # sending a message on the telegram channel
            HERA.send_message(
                'params: {}\n MRR is: {}\n\n'.format(params_dict, score))


if __name__ == "__main__":
    m = XGBoostWrapper(mode='small')
    v = RandomValidator(m)
    v.validate(100)
