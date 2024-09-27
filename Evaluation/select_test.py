import os
import pandas as pd
import random
import shutil


source_dir = '/root/autodl-tmp/celebAHQ/CelebA-HQ-256'
test_set_dir = '/root/autodl-tmp/celebAHQ/test_set'
attr_csv_path = '/root/autodl-tmp/celebAHQ/CelebAMask-HQ-attribute-anno.csv'
test_label_csv_path = '/root/autodl-tmp/celebAHQ/test_set_label.csv'

if not os.path.exists(test_set_dir):
    os.makedirs(test_set_dir)


all_images = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
selected_images = random.sample(all_images, 8500)


for image in selected_images:
    shutil.copy(os.path.join(source_dir, image), os.path.join(test_set_dir, image))


attr_df = pd.read_csv(attr_csv_path)


attr_df['Chubby'] = attr_df['Chubby'].replace(-1, 0)
attr_df['Big_Nose'] = attr_df['Big_Nose'].replace(-1, 0)
attr_df['Young'] = attr_df['Young'].replace(-1, 0)
attr_df['Mouth_Slightly_Open'] = attr_df['Mouth_Slightly_Open'].replace(-1, 0)
attr_df['Wearing_Lipstick'] = attr_df['Wearing_Lipstick'].replace(-1, 0)
attr_df['Blond_Hair'] = attr_df['Blond_Hair'].replace(-1, 0)


selected_image_ids = [img for img in selected_images]
filtered_attr_df = attr_df[attr_df['image_id'].isin(selected_image_ids)][['image_id', 'Chubby', 'Big_Nose', 'Young','Mouth_Slightly_Open','Wearing_Lipstick','Blond_Hair']]


filtered_attr_df.to_csv(test_label_csv_path, index=False)

print(f"Successfully selected 8500 images and saved their labels to {test_label_csv_path}")


# # # import os
# # # import csv

# # # file_paths = [
# # #     '/root/DiME-main/celebA_dataset/Dataprocess/0611celebA_original.csv',
# # #     '/root/DiME-main/celebA_dataset/Dataprocess/0613celebA_original.csv',
# # #     '/root/DiME-main/celebA_dataset/Dataprocess/0614celebA_original.csv'
# # # ]

# # # total_lines = 0

# # # for file_path in file_paths:
# # #     if os.path.exists(file_path):
# # #         with open(file_path, 'r') as f:
# # #             reader = csv.reader(f)
# # #             lines = sum(1 for row in reader)
# # #             print(f"{file_path}  {lines} ")
# # #             total_lines += lines

# # # print(f" {total_lines} ")
# import torch
# import pickle
# import os

# # Define the paths
# archive_dir = '/root/DiME-main/models/archive'
# new_model_path = '/root/DiME-main/models/classifier_celebAHQ.pth'

# # Read the files from the archive directory
# data_dir = os.path.join(archive_dir, 'data')
# data_pkl_path = os.path.join(archive_dir, 'data.pkl')
# version_path = os.path.join(archive_dir, 'version')

# # Function to load persistent IDs
# def persistent_load(saved_id):
#     assert isinstance(saved_id, tuple)
#     typename = saved_id[0]
#     if typename == 'storage':
#         data_type, root_key, location, size = saved_id[1:]
#         storage = torch.FloatStorage(size)
#         return storage
#     else:
#         raise RuntimeError("Unknown typename for persistent_load")

# # Load data.pkl with persistent_load function
# with open(data_pkl_path, 'rb') as f:
#     unpickler = pickle.Unpickler(f)
#     unpickler.persistent_load = persistent_load
#     data_pkl = unpickler.load()

# # Read version file (assuming it's a text file)
# with open(version_path, 'r') as f:
#     version = f.read().strip()

# # Load all files in the data directory
# data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]

# data = {}
# for file in data_files:
#     with open(file, 'rb') as f:
#         try:
#             data[file] = torch.load(f)
#         except Exception as e:
#             print(f"Skipping file {file} due to error: {e}")

# # Combine all information into a single dictionary
# model_data = {
#     'data': data,
#     'data_pkl': data_pkl,
#     'version': version
# }

# # Save the combined data into a new .pth file
# torch.save(model_data, new_model_path)

# print(f'Model saved to {new_model_path}')



# import os


# directory_path = '/root/autodl-tmp/output0708_CelebAHQ_Bignose_Young/Results/exp/name/CC/CCF/CF'


# image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# image_count = 0
# for root, dirs, files in os.walk(directory_path):
#     for file in files:
#         if any(file.lower().endswith(ext) for ext in image_extensions):
#             image_count += 1

# print(f'{directory_path} ：{image_count}')

