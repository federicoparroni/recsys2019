import utils.check_folder as cf
import time


class writer:
    def __init__(self, file_base_path, file_name):
        cf.check_folder(file_base_path)
        self.path = '{}/{}_{}'.format(file_base_path, time.strftime('%d_%b-%Hh-%Mm-%Ss'), file_name)

    def write(self, string_to_write):
        file = open(self.path, 'a')
        file.write(string_to_write)
        file.close()




