# import os
dirpath = 'C:/DL lecture slides/DeepD2C/video/test2'

import os
from pathlib import Path

paths = sorted(Path(dirpath).iterdir(), key=os.path.getmtime)

# files = os.listdir(path)
# files.sort(key=os.path.getctime)

def rename():
    for index, file in enumerate(paths):
        print('test')
        os.rename(os.path.join(dirpath, file), os.path.join(dirpath, ''.join(['90degree-10-cm', str(index + 1), '.png'])))

rename()