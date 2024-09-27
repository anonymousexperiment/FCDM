import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Define the directories containing the images
# image_dirA = '/root/attGAN/origin'
# image_dirB = '/root/attGAN/Smiling'


image_dirA = '/root/autodl-tmp/attGAN/celebAHQ/Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/origin'
image_dirB = '/root/autodl-tmp/attGAN/celebAHQ/Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/Wearing_Lipstick'

# Helper function to get the first 25,000 image paths from a directory
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

# Get the first 25,000 images from both directories
image_pathsA = get_first_n_images(image_dirA, n=17000)
image_pathsB = get_first_n_images(image_dirB, n=17000)

# Initialize a list to store SSIM scores
ssim_scores = []

# Loop through the selected images
for image_pathA, image_pathB in zip(image_pathsA, image_pathsB):
    # Load the images and convert them to grayscale
    imageA = cv2.imread(image_pathA, cv2.IMREAD_GRAYSCALE)
    imageB = cv2.imread(image_pathB, cv2.IMREAD_GRAYSCALE)
    
    # Resize images to the same dimensions if necessary
    if imageA.shape != imageB.shape:
        imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))

    # Calculate the SSIM between the two images
    similarity_index, _ = ssim(imageA, imageB, full=True)
    ssim_scores.append(similarity_index)

# Calculate the average SSIM score
average_ssim = np.mean(ssim_scores) if ssim_scores else float('nan')

print(f"Average SSIM: {average_ssim}")

# import os
# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity as ssim

# # Define the directories containing the images

# real_image_dirs = [
#     '/root/autodl-tmp/output0709_CelebAHQ_Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/Original/Correct',
#     '/root/autodl-tmp/output0710_CelebAHQ_Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/Original/Correct'
# ]
# generated_image_dirs = [
#     '/root/autodl-tmp/output0709_CelebAHQ_Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/Results/exp/name/CC/CCF/CF',
#     '/root/autodl-tmp/output0710_CelebAHQ_Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/Results/exp/name/CC/CCF/CF'
# ]
# # Initialize a list to store SSIM scores
# ssim_scores = []

# # Function to get all image paths from multiple directories
# def get_all_image_paths(directories):
#     all_image_paths = []
#     for directory in directories:
#         for root, _, files in os.walk(directory):
#             for file in files:
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#                     all_image_paths.append(os.path.join(root, file))
#     return all_image_paths

# # Get all image paths for real and generated images
# real_image_paths = get_all_image_paths(real_image_dirs)
# generated_image_paths = get_all_image_paths(generated_image_dirs)

# # Create a set of image names to ensure matching images between directories
# real_image_names = set(os.path.basename(path) for path in real_image_paths)
# generated_image_names = set(os.path.basename(path) for path in generated_image_paths)
# common_image_names = real_image_names.intersection(generated_image_names)

# # Calculate SSIM scores for matching images
# for image_name in common_image_names:
#     image_pathA = next(path for path in real_image_paths if os.path.basename(path) == image_name)
#     image_pathB = next(path for path in generated_image_paths if os.path.basename(path) == image_name)
    
#     # Load the images and convert them to grayscale
#     imageA = cv2.imread(image_pathA, cv2.IMREAD_GRAYSCALE)
#     imageB = cv2.imread(image_pathB, cv2.IMREAD_GRAYSCALE)
    
#     # Resize images to the same dimensions if necessary
#     if imageA.shape != imageB.shape:
#         imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))

#     # Calculate the SSIM between the two images
#     similarity_index, _ = ssim(imageA, imageB, full=True)
#     ssim_scores.append(similarity_index)

# # Calculate the average SSIM score
# average_ssim = np.mean(ssim_scores) if ssim_scores else float('nan')

# print(f"Average SSIM: {average_ssim}")




# import os
# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity as ssim
# import pandas as pd

# # Define the directories containing the images
# real_images_folder = '/root/DiME-main/celebA_dataset/img_align_celeba'
# csv_file = '/root/CFDM/Ablation_experiment/recon_loss_cf.csv'

# # Helper function to get specific image paths from a directory
# def get_specific_images(directory, start_index=60001, end_index=67300):
#     return [os.path.join(directory, f"{i:06d}.jpg") for i in range(start_index, end_index + 1)]

# # Read image paths from the CSV file
# def get_image_paths_from_csv(csv_file):
#     df = pd.read_csv(csv_file)
#     return df['bestCF'].tolist()

# # Get the specific images from real_images_folder
# image_pathsA = get_specific_images(real_images_folder)

# # Get the images from the CSV file for generated_images_folder
# image_pathsB = get_image_paths_from_csv(csv_file)

# # Ensure both directories have the same number of images
# min_length = min(len(image_pathsA), len(image_pathsB))
# image_pathsA = image_pathsA[:min_length]
# image_pathsB = image_pathsB[:min_length]

# # Initialize a list to store SSIM scores
# ssim_scores = []

# # Loop through the selected images
# for image_pathA, image_pathB in zip(image_pathsA, image_pathsB):
#     # Load the images and convert them to grayscale
#     imageA = cv2.imread(image_pathA, cv2.IMREAD_GRAYSCALE)
#     imageB = cv2.imread(image_pathB, cv2.IMREAD_GRAYSCALE)
    
#     # Resize images to the same dimensions if necessary
#     if imageA.shape != imageB.shape:
#         imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))

#     # Calculate the SSIM between the two images
#     similarity_index, _ = ssim(imageA, imageB, full=True)
#     ssim_scores.append(similarity_index)

# # Calculate the average SSIM score
# average_ssim = np.mean(ssim_scores) if ssim_scores else float('nan')

# print(f"Average SSIM: {average_ssim}")
