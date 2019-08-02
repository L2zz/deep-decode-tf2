"""
module for get file list from target dir
"""
import os

def file_from_dir(target_dir):
    """
    @param
        target_dir: target directory name(path)
    @return
        file_list: list of files in target_dir
                   reprensented with <target dir>/<file name>
    """
    file_list = os.listdir(target_dir)
    file_list.sort()

    # represent target directory
    file_list = [target_dir + "/" + file for file in file_list]

    return file_list

if __name__ == "__main__":
    DIR = "org"
    print(file_from_dir(DIR))
