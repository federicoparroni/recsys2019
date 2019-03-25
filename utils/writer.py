import utils.check_folder as cf

def write(file_base_path, file_name, string_to_write):
    cf.check_folder(file_base_path)
    file = open('{}/{}'.format(file_base_path, file_name), 'a')
    file.write(string_to_write)
    file.close()


if __name__ == '__main__':
    write('validation_result', 'prova', 'cazzo')

