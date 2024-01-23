import os
import random
import shutil

raw_folder = 'dataset/wifi/raw'
cond_folder = 'dataset/wifi/cond'

files = os.listdir(raw_folder)

random_files = random.sample(files, 1500)

os.makedirs(cond_folder, exist_ok=True)

for file in random_files:
    src_path = os.path.join(raw_folder, file)
    dst_path = os.path.join(cond_folder, file)
    shutil.move(src_path, dst_path)
