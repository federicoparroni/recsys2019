from recommenders.XGBoost import XGBoostWrapper
from recommenders.XGBoost import XGBoostWrapperSmartValidation
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
    convention: range in tuple -> consider all the possible value in between
                range in list  -> consider just the values in the list
    """

    def __init__(self, reference_object, reference_object_for_sub_exporter=None, granularity=100, automatic_export=True, user='default'):
        self.reference_object = reference_object
        self.granularity = granularity
        self.user = user

        # used to know where to store the validation result
        self.file_base_path = 'validation_result'
        self.file_name = str(reference_object.__class__).split(
            '.')[-1].replace('\'>', '')
        self.writer = writer(
            file_base_path=self.file_base_path, file_name=self.file_name)

        self.automatic_export = None
        if automatic_export:
            if reference_object_for_sub_exporter is None:
                self.automatic_export = AutomaticSubExporter(reference_object, self.user)
            else:
                self.automatic_export = AutomaticSubExporter(reference_object_for_sub_exporter, self.user)

    def sample(self, obj):
        if type(obj) == tuple:
            low_bound = obj[0]
            upp_bound = obj[1]
            rand_numb = randint(0, self.granularity)
            step = (upp_bound - low_bound)/self.granularity
            return low_bound + step*rand_numb
        elif type(obj) == list:
            return obj[randint(0, len(obj)-1)]

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

            # assign it again in case a fixed parameter has changed
            self.fixed_params_dict, self.hyperparameters_dict = model.get_params()
            params_dict = {**self.fixed_params_dict, **sampled_params}
            if self.automatic_export != None:
                self.automatic_export.check_if_export(score, params_dict)

            self.writer.write(
                'params: {}\n MRR is: {}\n\n'.format(params_dict, score))

            # sending a message on the telegram channel
            HERA.send_message(
               'name: {} params: {}\n MRR is: {}\n\n'.format(model.name, params_dict, score), self.user)
            print('name: {} params: {}\n MRR is: {}\n\n'.format(model.name, params_dict, score))

if __name__ == "__main__":
    from utils.menu import cluster_selection
    from utils.menu import mode_selection
    from utils.menu import single_choice
    mode = mode_selection()
    cluster = cluster_selection()
    kind = input('insert the kind: ')
    m = XGBoostWrapperSmartValidation(mode=mode, cluster=cluster, kind=kind, ask_to_load=False)
    a = XGBoostWrapper(mode='full', cluster=cluster, kind=kind)
    v = RandomValidator(m, automatic_export=False, reference_object_for_sub_exporter=False, user='edo')
    v.validate(100)
