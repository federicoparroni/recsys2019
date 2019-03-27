import os


def check_folder(path):
    if not os.path.exists(path):
        print(f'{path} folder created')
        os.makedirs(path, exist_ok=True)
