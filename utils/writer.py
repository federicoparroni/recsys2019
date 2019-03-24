import utils.check_folder as cf

class Writer:

    def __init__(self, file_base_path, file_name):
        cf.check_folder(file_base_path)
        self.file = open('{}/{}'.format(file_base_path, file_name), 'w+')

    def write_line(self, string_to_write):
        self.file.write(string_to_write)


if __name__ == '__main__':
    w = Writer('validation_result', 'prova.txt')
    for i in range(10):
        w.write_line('cazzo\n')
