import os
import torch
from pytorch_fid import fid_score

# Define the paths to the directories containing real and generated images
real_images_folder = '/root/autodl-tmp/attGAN/celebAHQ/Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/origin'
generated_images_folder = '/root/autodl-tmp/attGAN/celebAHQ/Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/Wearing_Lipstick'

# Helper function to get the first 25000 image paths from a directory
def get_first_n_images(directory, n=17000):
    all_images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                all_images.append(os.path.join(root, file))
            if len(all_images) >= n:
                break
        if len(all_images) >= n:
            break
    return all_images[:n]

# Get the first 25000 images from both directories
real_image_paths = get_first_n_images(real_images_folder, n=23000)
generated_image_paths = get_first_n_images(generated_images_folder, n=23000)

# Write the paths to temporary text files (required by fid_score.calculate_fid_given_paths)
real_images_dir = '/root/Young_sens_Male/real_images'
generated_images_dir = '/root/Young_sens_Male/generated_images'

os.makedirs(real_images_dir, exist_ok=True)
os.makedirs(generated_images_dir, exist_ok=True)

# Create symlinks for real images
for i, path in enumerate(real_image_paths):
    target_path = os.path.join(real_images_dir, f'image_{i}.png')
    if os.path.exists(target_path):
        os.remove(target_path)
    os.symlink(path, target_path)

# Create symlinks for generated images
for i, path in enumerate(generated_image_paths):
    target_path = os.path.join(generated_images_dir, f'image_{i}.png')
    if os.path.exists(target_path):
        os.remove(target_path)
    os.symlink(path, target_path)

# Calculate the FID score
fid_value = fid_score.calculate_fid_given_paths([real_images_dir, generated_images_dir],
                                                batch_size=50,
                                                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                                dims=2048)

print('FID value:', fid_value)



# Clean up temporary directories
import shutil
shutil.rmtree(real_images_dir)
shutil.rmtree(generated_images_dir)





# import os
# import torch
# from pytorch_fid import fid_score

# # Define image directories


# real_images_dirs = [
#     '/root/autodl-tmp/output0706_CelebAHQ_Bignose_Young/Original/Correct',
#     '/root/autodl-tmp/output0708_CelebAHQ_Bignose_Young/Original/Correct'
# ]
# generated_image_dirs = [
#     '/root/autodl-tmp/output0706_CelebAHQ_Bignose_Young/Results/exp/name/CC/CCF/CF',
#     '/root/autodl-tmp/output0708_CelebAHQ_Bignose_Young/Results/exp/name/CC/CCF/CF'
# ]
# # Function to get all image paths from multiple directories
# def get_all_image_paths(directories):
#     all_image_paths = []
#     for directory in directories:
#         for root, _, files in os.walk(directory):
#             for file in files:
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#                     all_image_paths.append(os.path.join(root, file))
#     return all_image_paths

# # Combine image directories into a single list for real and generated images
# real_image_paths = get_all_image_paths(real_images_dirs)
# generated_image_paths = get_all_image_paths(generated_image_dirs)

# # Define temporary directories to hold symlinks
# real_images_dir = '/root/autodl-tmp/CFDM1_Eyeglasses_real_images_total'
# generated_images_dir = '/root/autodl-tmp/CFDM1_Eyeglasses_generated_images_total'

# # Create the temporary directories
# os.makedirs(real_images_dir, exist_ok=True)
# os.makedirs(generated_images_dir, exist_ok=True)

# # Create symlinks for real images
# for i, path in enumerate(real_image_paths):
#     os.symlink(path, os.path.join(real_images_dir, f'image_{i}.png'))

# # Create symlinks for generated images
# for i, path in enumerate(generated_image_paths):
#     os.symlink(path, os.path.join(generated_images_dir, f'image_{i}.png'))

# # Calculate the FID score
# fid_value = fid_score.calculate_fid_given_paths([real_images_dir, generated_images_dir],
#                                                 batch_size=50,
#                                                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#                                                 dims=2048)

# print('FID value:', fid_value)


##################################### Ablation_experiment ##################################### 

# import os
# import torch
# from pytorch_fid import fid_score
# import pandas as pd

# # Define the paths to the directories containing real and generated images
# real_images_folder = '/root/DiME-main/celebA_dataset/img_align_celeba'
# csv_file = '/root/CFDM/Ablation_experiment/recon_loss_cf.csv'

# # Helper function to get the specific image paths from a directory
# def get_specific_images(directory, start_index=60001, end_index=67300):
#     return [os.path.join(directory, f"{i:06d}.jpg") for i in range(start_index, end_index + 1)]

# # Read image paths from the CSV file
# def get_image_paths_from_csv(csv_file):
#     df = pd.read_csv(csv_file)
#     return df['bestCF'].tolist()

# # Get the specific images from real_images_folder
# real_image_paths = get_specific_images(real_images_folder)

# # Get the images from the CSV file for generated_images_folder
# generated_image_paths = get_image_paths_from_csv(csv_file)

# # Ensure both directories have the same number of images
# min_length = min(len(real_image_paths), len(generated_image_paths))
# real_image_paths = real_image_paths[:min_length]
# generated_image_paths = generated_image_paths[:min_length]

# # Write the paths to temporary text files (required by fid_score.calculate_fid_given_paths)
# real_images_dir = '/root/Ablation_experiment/real_images'
# generated_images_dir = '/root/Ablation_experiment/generated_images'

# os.makedirs(real_images_dir, exist_ok=True)
# os.makedirs(generated_images_dir, exist_ok=True)

# # Create symlinks for real images
# for i, path in enumerate(real_image_paths):
#     target_path = os.path.join(real_images_dir, f'image_{i}.png')
#     if os.path.exists(target_path):
#         os.remove(target_path)
#     os.symlink(path, target_path)

# # Create symlinks for generated images
# for i, path in enumerate(generated_image_paths):
#     target_path = os.path.join(generated_images_dir, f'image_{i}.png')
#     if os.path.exists(target_path):
#         os.remove(target_path)
#     os.symlink(path, target_path)

# # Calculate the FID score
# fid_value = fid_score.calculate_fid_given_paths([real_images_dir, generated_images_dir],
#                                                 batch_size=50,
#                                                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#                                                 dims=2048)

# print('FID value:', fid_value)
