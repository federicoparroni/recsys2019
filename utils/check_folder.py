import os


def check_folder(path):
    split_folder = os.path.split(path)
    if '.' in split_folder[1]:
        # path is a file
        path = split_folder[0]
    if not os.path.exists(path):
        print(f'{path} folder created')
        os.makedirs(path, exist_ok=True)
