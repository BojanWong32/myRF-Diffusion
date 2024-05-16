import os
import shutil

# raw_folder = 'D:/BaiduNetdiskDownload/DFSExtractionCode/DFSExtractionCode/Output/20181109/user1'
# cond_folder = 'dataset/wifi/raw'
#
# os.makedirs(cond_folder, exist_ok=True)

# for file in os.listdir(raw_folder):
#     repetition = int(file.split('-')[4])
#     if repetition <= 5:
#         src_path = os.path.join(raw_folder, file)
#         dst_path = os.path.join(cond_folder, file)
#         shutil.copy(src_path, dst_path)
raw_folder = 'dataset/wifi/raw'
cond_folder = 'dataset/wifi/cond'

os.makedirs(cond_folder, exist_ok=True)

file_list = os.listdir(raw_folder)
selected_files = file_list[:1500]  # 选择前1500个文件

for file in selected_files:
    src_path = os.path.join(raw_folder, file)
    dst_path = os.path.join(cond_folder, file)
    shutil.copy(src_path, dst_path)