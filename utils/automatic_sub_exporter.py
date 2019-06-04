# from multiprocessing import Process
from threading import Thread
import out
import utils.telegram_bot as HERA
from utils.check_folder import check_folder
import time
import numpy as np


class AutomaticSubExporter:

    def __init__(self, referenced_object, user='default'):
        self.reference_object = referenced_object
        self.threshold = 1e-4
        self.actual_score = 0
        self.user = user

    def export(self, obj, params_dict, mode, mrr):
        params_dict['mode'] = mode
        instance = obj(**params_dict)
        # print('EXPORTING sub and scores algo {} with score {} in mode {} with params {}'.format(instance.name, mrr, mode, params_dict))
        HERA.send_message('EXPORTING sub and scores algo {} with score {} in mode {} with params {}'.format(instance.name, mrr, mode, params_dict), self.user)
        instance.run(export_sub=True, export_scores=True)

    def check_if_export(self, mrr, params_dict):
        if self.actual_score == 0:
            self.actual_score = mrr
        else:
            if mrr >= self.actual_score + self.threshold:
                l = Thread(target=self.export, args=(self.reference_object.__class__, params_dict, 'local', mrr))
                l.start()
                f = Thread(target=self.export, args=(self.reference_object.__class__, params_dict, 'full', mrr))
                f.start()
                self.actual_score = mrr
