
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import lpips

# Initialize the LPIPS model
lpips_model = lpips.LPIPS(net='alex').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='alex')

# Define image directories

image_dir1  = '/root/autodl-tmp/attGAN/celebAHQ/Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/origin'
image_dir2 = '/root/autodl-tmp/attGAN/celebAHQ/Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/Wearing_Lipstick'
# Preprocess the images
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Helper function to get the first 25,000 image paths from a directory
def get_first_n_images(directory, n=23000):
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
image_paths1 = get_first_n_images(image_dir1, n=23000)
image_paths2 = get_first_n_images(image_dir2, n=23000)

# Initialize a list to store LPIPS scores
lpips_scores = []

# Loop through the selected images
for image_path1, image_path2 in zip(image_paths1, image_paths2):
    # Load and preprocess the images
    image1 = Image.open(image_path1).convert('RGB')
    image2 = Image.open(image_path2).convert('RGB')
    
    image1 = preprocess(image1).unsqueeze(0)
    image2 = preprocess(image2).unsqueeze(0)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        image1 = image1.cuda()
        image2 = image2.cuda()
    
    # Calculate the LPIPS similarity
    similarity_score = lpips_model(image1, image2)
    lpips_scores.append(similarity_score.item())

# Calculate the average LPIPS score
average_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else float('nan')

print(f"Average LPIPS Similarity: {average_lpips}")









# import os
# import torch
# import lpips
# from PIL import Image
# from torchvision import transforms

# # Initialize the LPIPS model
# lpips_model = lpips.LPIPS(net='alex').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='alex')

# # Define image directories
# image_dirs1 = [
#     '/root/autodl-tmp/output0709_CelebAHQ_Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/Original/Correct',
#     '/root/autodl-tmp/output0710_CelebAHQ_Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/Original/Correct'
# ]
# image_dirs2 = [
#     '/root/autodl-tmp/output0709_CelebAHQ_Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/Results/exp/name/CC/CCF/CF',
#     '/root/autodl-tmp/output0710_CelebAHQ_Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/Results/exp/name/CC/CCF/CF'
# ]

# # Preprocess the images
# preprocess = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
# ])

# # Initialize a list to store LPIPS scores
# lpips_scores = []

# # Function to get all image paths from multiple directories
# def get_all_image_paths(directories):
#     all_image_paths = []
#     for directory in directories:
#         for root, _, files in os.walk(directory):
#             for file in files:
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#                     all_image_paths.append(os.path.join(root, file))
#     return all_image_paths

# # Get all image paths from the directories
# all_image_paths1 = get_all_image_paths(image_dirs1)
# all_image_paths2 = get_all_image_paths(image_dirs2)

# # Create a set of image names to ensure matching images between directories
# image_names1 = set(os.path.basename(path) for path in all_image_paths1)
# image_names2 = set(os.path.basename(path) for path in all_image_paths2)
# common_image_names = image_names1.intersection(image_names2)

# # Calculate LPIPS scores for matching images
# for image_name in common_image_names:
#     image_path1 = next(path for path in all_image_paths1 if os.path.basename(path) == image_name)
#     image_path2 = next(path for path in all_image_paths2 if os.path.basename(path) == image_name)
    
#     # Load and preprocess the images
#     image1 = Image.open(image_path1).convert('RGB')
#     image2 = Image.open(image_path2).convert('RGB')
    
#     image1 = preprocess(image1).unsqueeze(0)
#     image2 = preprocess(image2).unsqueeze(0)
    
#     # Move to GPU if available
#     if torch.cuda.is_available():
#         image1 = image1.cuda()
#         image2 = image2.cuda()
    
#     # Calculate the LPIPS similarity
#     similarity_score = lpips_model(image1, image2)
#     lpips_scores.append(similarity_score.item())

# # Calculate the average LPIPS score
# average_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else float('nan')

# print(f"Average LPIPS Similarity: {average_lpips}")


# ##################################### Ablation_experiment ##################################### 
# import os
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import lpips
# import pandas as pd

# # Initialize the LPIPS model
# lpips_model = lpips.LPIPS(net='alex').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='alex')

# # Define image directories
# image_dir1 = '/root/DiME-main/celebA_dataset/img_align_celeba'
# csv_file = '/root/CFDM/Ablation_experiment/recon_loss_cf.csv'

# # Preprocess the images
# preprocess = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
# ])

# # Helper function to get the first 25,000 specific image paths from a directory
# def get_specific_images(directory, start_index=60001, end_index=67300):
#     return [os.path.join(directory, f"{i:06d}.jpg") for i in range(start_index, end_index + 1)]

# # Read image paths from the CSV file
# def get_image_paths_from_csv(csv_file):
#     df = pd.read_csv(csv_file)
#     return df['bestCF'].tolist()

# # Get the specific images from image_dir1
# image_paths1 = get_specific_images(image_dir1)

# # Get the images from the CSV file for image_dir2
# image_paths2 = get_image_paths_from_csv(csv_file)

# # Ensure both directories have the same number of images
# min_length = min(len(image_paths1), len(image_paths2))
# image_paths1 = image_paths1[:min_length]
# image_paths2 = image_paths2[:min_length]

# # Initialize a list to store LPIPS scores
# lpips_scores = []

# # Loop through the selected images
# for image_path1, image_path2 in zip(image_paths1, image_paths2):
#     # Load and preprocess the images
#     image1 = Image.open(image_path1).convert('RGB')
#     image2 = Image.open(image_path2).convert('RGB')
    
#     image1 = preprocess(image1).unsqueeze(0)
#     image2 = preprocess(image2).unsqueeze(0)
    
#     # Move to GPU if available
#     if torch.cuda.is_available():
#         image1 = image1.cuda()
#         image2 = image2.cuda()
    
#     # Calculate the LPIPS similarity
#     similarity_score = lpips_model(image1, image2)
#     lpips_scores.append(similarity_score.item())

# # Calculate the average LPIPS score
# average_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else float('nan')

# print(f"Average LPIPS Similarity: {average_lpips}")
