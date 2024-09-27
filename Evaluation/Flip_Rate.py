
# import os
# import re

# # info_dirs = [
# #     '/root/autodl-tmp/output0611_CelebA/Results/exp/name/CC/CCF/Info',
# #     #'/root/autodl-tmp/output0611_CelebA/Results/exp/name/CC/ICF/Info',
# #     '/root/autodl-tmp/output0611_CelebA/Results/exp/name/IC/CCF/Info',
# #     # '/root/autodl-tmp/output0611_CelebA/Results/exp/name/IC/ICF/Info',
# #     '/root/autodl-tmp/output0613_CelebA/Results/exp/name/CC/CCF/Info',
# #     #'/root/autodl-tmp/output0613_CelebA/Results/exp/name/CC/ICF/Info',
# #     '/root/autodl-tmp/output0613_CelebA/Results/exp/name/IC/CCF/Info',
# #     # '/root/autodl-tmp/output0613_CelebA/Results/exp/name/IC/ICF/Info',
# #     '/root/autodl-tmp/output0614_CelebA/Results/exp/name/CC/CCF/Info',
# #     #'/root/autodl-tmp/output0614_CelebA/Results/exp/name/CC/ICF/Info',
# #     '/root/autodl-tmp/output0614_CelebA/Results/exp/name/IC/CCF/Info',
# #     # '/root/autodl-tmp/output0614_CelebA/Results/exp/name/IC/ICF/Info',
# #     '/root/autodl-tmp/output0615_CelebA/Results/exp/name/CC/CCF/Info',
# #     #'/root/autodl-tmp/output0615_CelebA/Results/exp/name/CC/ICF/Info',
# #     '/root/autodl-tmp/output0615_CelebA/Results/exp/name/IC/CCF/Info',
# #     # '/root/autodl-tmp/output0615_CelebA/Results/exp/name/IC/ICF/Info',
# #     '/root/DiME-main/output0616_CelebA/Results/exp/name/CC/CCF/Info',
# #     #'/root/DiME-main/output0616_CelebA/Results/exp/name/CC/ICF/Info',
# #     '/root/DiME-main/output0616_CelebA/Results/exp/name/IC/CCF/Info',
# #     # '/root/DiME-main/output0616_CelebA/Results/exp/name/IC/ICF/Info',
# #     '/root/DiME-main/output0617_CelebA/Results/exp/name/CC/CCF/Info',
# #     #'/root/DiME-main/output0617_CelebA/Results/exp/name/CC/ICF/Info',
# #     '/root/DiME-main/output0617_CelebA/Results/exp/name/IC/CCF/Info',
# #     # '/root/DiME-main/output0617_CelebA/Results/exp/name/IC/ICF/Info'
# # ]
# info_dirs = [
#     '/root/autodl-tmp/output0709_CelebAHQ_Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/Results/exp/name/CC/CCF/Info',
#     '/root/autodl-tmp/output0709_CelebAHQ_Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/Results/exp/name/IC/CCF/Info',
#     '/root/autodl-tmp/output0710_CelebAHQ_Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/Results/exp/name/CC/CCF/Info',
#     '/root/autodl-tmp/output0710_CelebAHQ_Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/Results/exp/name/IC/CCF/Info'
# ]

# # Initialize counters
# total_files = 0
# successful_flips_first_element = 0
# successful_flips_second_element = 0

# # Loop through each directory
# for info_dir in info_dirs:
#     # Loop through each .txt file in the directory
#     for file_name in os.listdir(info_dir):
#         if file_name.endswith('.txt'):
#             file_path = os.path.join(info_dir, file_name)
#             with open(file_path, 'r') as file:
#                 # Read and parse the contents of the file
#                 contents = file.readlines()
#                 label_line = contents[0].strip()  # label: [0 1]
#                 cf_pred_line = contents[3].strip()  # cf pred: [1 0]

#                 # Extract the first elements of label and cf pred
#                 label_first_element = int(label_line.split('[')[1].split(']')[0].split()[0])  # Extract the first element
#                 cf_pred_first_element = int(cf_pred_line.split('[')[1].split(']')[0].split()[0])  # Extract the first element

#                 # Extract the second elements of label and cf pred
#                 label_second_element = int(label_line.split('[')[1].split(']')[0].split()[1])  # Extract the second element
#                 cf_pred_second_element = int(cf_pred_line.split('[')[1].split(']')[0].split()[1])  # Extract the second element

#                 # Check if the flip is successful for the first element
#                 if label_first_element != cf_pred_first_element:
#                     successful_flips_first_element += 1

#                 # Check if the flip is successful for the second element
#                 if label_second_element != cf_pred_second_element:
#                     successful_flips_second_element += 1

#                 total_files += 1

# # Calculate the flip rates
# flip_rate_first_element = successful_flips_first_element / total_files if total_files > 0 else 0
# flip_rate_second_element = successful_flips_second_element / total_files if total_files > 0 else 0

# print(f"Total files: {total_files}")
# print(f"Successful flips for Male: {successful_flips_first_element}")
# print(f"Flip rate for Male: {flip_rate_first_element:.2f}")
# print(f"Successful flips for Smiling: {successful_flips_second_element}")
# print(f"Flip rate for Smiling: {flip_rate_second_element:.2f}")







######FR############
import pandas as pd


original_csv = '/root/autodl-tmp/Vikram_et_al./celebAHQ/celebAHQ_Mouth_Sligntly_Open_Wearing_Lipstick_Blond_Hair/label_origin.csv'
cf_male_csv = '/root/autodl-tmp/Vikram_et_al./celebAHQ/celebAHQ_Mouth_Sligntly_Open_Wearing_Lipstick_Blond_Hair/label_Mouth_Slightly_Open.csv'


original_df = pd.read_csv(original_csv)
cf_male_df = pd.read_csv(cf_male_csv)


cf_male_df_25000 = cf_male_df.head(23000)


# if 'Big_Nose' not in original_df.columns or 'Big_Nose' not in cf_male_df_25000.columns:
#     raise ValueError("CSV lose 'Big_Nose' ")


total_rows = min(len(original_df), len(cf_male_df_25000))
successful_reversals = 0

for i in range(total_rows):
    if original_df.at[i, 'Mouth_Slightly_Open'] != cf_male_df_25000.at[i, 'Mouth_Slightly_Open']:#smiling
        successful_reversals += 1

reversal_success_rate = successful_reversals / total_rows if total_rows > 0 else 0

print(f"Total rows checked: {total_rows}")
print(f"Successful reversals: {successful_reversals}")
print(f"Reversal success rate: {reversal_success_rate:.2f}")


#####FRY############
import pandas as pd

# original_csv = '/root/attGAN/labels_original.csv'
# cf_smile_csv = '/root/attGAN/labels_Smiling.csv'




original_csv = '/root/autodl-tmp/Vikram_et_al./celebAHQ/celebAHQ_Mouth_Sligntly_Open_Wearing_Lipstick_Blond_Hair/label_origin.csv'
cf_smile_csv = '/root/autodl-tmp/Vikram_et_al./celebAHQ/celebAHQ_Mouth_Sligntly_Open_Wearing_Lipstick_Blond_Hair/label_Mouth_Slightly_Open.csv'


original_df = pd.read_csv(original_csv)
cf_smile_df = pd.read_csv(cf_smile_csv)


cf_smile_df_25000 = cf_smile_df.head(23000)


# required_columns = ['Male', 'Young']
# if not all(column in original_df.columns for column in required_columns):
#     raise ValueError(" CSV lose")
# if 'Male' not in cf_smile_df_25000.columns or 'Young' not in cf_smile_df_25000.columns:
#     raise ValueError("CF Smile CSV ")


total_rows = min(len(original_df), len(cf_smile_df_25000))
successful_reversals = 0

for i in range(total_rows):
    if (original_df.at[i, 'Mouth_Slightly_Open'] != cf_smile_df_25000.at[i, 'Mouth_Slightly_Open'] and 
        original_df.at[i, 'Blond_Hair'] == cf_smile_df_25000.at[i, 'Blond_Hair']):
        successful_reversals += 1

reversal_success_rate = successful_reversals / total_rows if total_rows > 0 else 0

print(f"Total rows checked: {total_rows}")
print(f"Successful reversals: {successful_reversals}")
print(f"Reversal success rate: {reversal_success_rate:.2f}")