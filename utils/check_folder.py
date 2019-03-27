import os


def check_folder(path):
    if os.path.isfile(path):
        path = os.path.split(path)[0]
    if not os.path.exists(path):
        print(f'{path} folder created')
        os.makedirs(path, exist_ok=True)
