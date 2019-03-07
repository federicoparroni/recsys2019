import os
import time

def create_folder(rootpath, filename, extension, append_time_suffix=True, date_format='%d-%m-%Y'):
    """
    Create a folder that is named as the current date for the specified file.

    Parameters
    ----------
    rootpath: (str) path of the folder where to create the new folder
    filename: (str) name of the file without extension
    extension : (str) extension of the file (without ".")
    append_time_suffix: (bool) if true, concatenate the current time at the end of the file
    date_format: (str) format of the date

    Returns
    -------
    (str) complete path of the file to be created inside the folder hierarchy
    """

    folder = time.strftime(date_format)
    filepath = '{}/{}/{}_{}.{}'.format(rootpath, folder, filename, time.strftime('%H-%M-%S') if append_time_suffix else '', extension)
    # create dir if not exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    return filepath