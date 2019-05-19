from multiprocessing import Process
import out
import utils.telegram_bot as HERA

class AutomaticSubExporter:

    def __init__(self, referenced_object):
        self.reference_object = referenced_object
        self.threshold = 1e-3
        self.actual_score = 0

    def export(self, obj, params_dict, mode, mrr):
        params_dict['mode'] = mode
        instance = obj(**params_dict)
        HERA.send_message('EXPORTING sub algo {} with score {} in mode {} with params {}'.format(instance.name, mrr, mode, params_dict))
        instance.fit()
        recommendations = instance.recommend_batch()
        out.create_sub(recommendations, submission_name='{}_{}_{}'.format(instance.name, mrr, mode))
        HERA.send_message('EXPORTED sub algo {} with score {} in mode {} with params {}'.format(instance.name, mrr, mode, params_dict))

    def check_if_export(self, mrr, params_dict):
        if self.actual_score == 0:
            self.actual_score = mrr
        else:
            if mrr >= self.actual_score + self.threshold:
                p = Process(target=self.export, args=(self.reference_object.__class__, params_dict, 'full', mrr))
                p.start()
                p = Process(target=self.export, args=(self.reference_object.__class__, params_dict, 'local', mrr))
                p.start()
                self.actual_score = mrr
