# import os
dirpath = r'C:/DL lecture slides/DeepD2C/ber_test_screen_brightness/reference_images'

import os
import re

file_list = os.listdir(dirpath)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


file_list.sort(key=natural_keys)

print(file_list)

def rename():
    for index, file in enumerate(file_list):
        print('test')
        os.rename(os.path.join(dirpath, file), os.path.join(dirpath, ''.join([ str(index + 1), '.jpg'])))

rename()
