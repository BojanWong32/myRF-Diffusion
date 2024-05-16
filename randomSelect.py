import os
import random
import shutil

raw_folder = 'dataset/wifi/dataset_all'
cond_folder = 'dataset/wifi/test_dataset2'

files = os.listdir(raw_folder)

random_files = random.sample(files, 4494)

os.makedirs(cond_folder, exist_ok=True)

for file in random_files:
    file_name = os.path.basename(file)
    receiver = int(file_name.split('r')[2].split('.')[0])
    # print(receiver)

    if receiver != 1:  # 只接受r1
        continue
    src_path = os.path.join(raw_folder, file)
    dst_path = os.path.join(cond_folder, file)
    shutil.copy(src_path, dst_path)
