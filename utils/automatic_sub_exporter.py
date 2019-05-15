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
        obj = self.reference_object.__class__(**params_dict)
        HERA.send_message('exporting sub algo {} with score {} in mode {}'.format(obj.name, mrr, mode))
        obj.fit()
        recommendations = obj.recommend_batch()
        out.create_sub(recommendations, submission_name='{}_{}_{}'.format(obj.name, mrr, mode))

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
                HERA.send_message(
                    'name: {} params: {}\n MRR is: {}\n\n'.format('sub exported!'))
