

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm

# Define paths
base_folder = '/root/DiME-main/celebA_dataset/img_align_celeba'
folder_A_base = '/root/autodl-tmp/output0617_CelebA/x_t/exp/name'
folder_B_base = '/root/DiME-main/output0617_2CelebA/x_t/exp/name'
folder_C_base = '/root/DiME-main/output0617_3CelebA/x_t/exp/name'
output_file = '/root/DiME-main/information_cf.csv'

# Load image and resize to 256x256
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((256, 256))  # Resize to 256x256
    image_tensor = torch.tensor(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    return image_tensor

# Calculate cosine similarity loss#loss值越小，越相似
def calculate_cosine_similarity(image1, image2):
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    similarity = torch.dot(image1_flat, image2_flat) / (torch.norm(image1_flat) * torch.norm(image2_flat))
    cosine_similarity_loss = 1 - similarity
    return cosine_similarity_loss.item()

# Define loss functions
def diversity_loss(images):
    total_loss = 0.0
    K = len(images)
    for i in range(K):
        for j in range(i + 1, K):
            cos_sim = calculate_cosine_similarity(images[i], images[j])
            total_loss += max(0, 0.05 - cos_sim)  # Adjusted threshold for diversity loss
    total_loss = 2/(K * (K - 1))  * total_loss
    return total_loss

def similarity_loss(original, images):
    min_loss = float('inf')
    best_img = None
    for img in images:
        cos_sim = calculate_cosine_similarity(original, img)
        if cos_sim < min_loss:
            min_loss = cos_sim
            best_img = img
    return min_loss, best_img

# Convert a tensor to a string identifier
def tensor_to_str(tensor):
    return str(tensor.numpy().flatten().tolist())

def find_best_counterfactuals(num_samples=20):
    results = []

    for i in tqdm(range(1, num_samples + 1), desc="Processing samples"):
        original_image_path = os.path.join(base_folder, f'06{i:04d}.jpg')
        original_image = load_image(original_image_path)

        folder_A = os.path.join(folder_A_base, f'{i - 1:06d}')
        folder_B = os.path.join(folder_B_base, f'{i - 1:06d}')
        folder_C = os.path.join(folder_C_base, f'{i - 1:06d}')

        filenames = ['0035.jpg', '0036.jpg', '0037.jpg', '0038.jpg', '0039.jpg']
        
        cf_candidates_A = [load_image(os.path.join(folder_A, fn)) for fn in filenames]
        cf_candidates_B = [load_image(os.path.join(folder_B, fn)) for fn in filenames]
        cf_candidates_C = [load_image(os.path.join(folder_C, fn)) for fn in filenames]

        best_combination = None
        min_total_loss = float('inf')
        best_cf_path = None

        for img_a in cf_candidates_A:
            for img_b in cf_candidates_B:
                for img_c in cf_candidates_C:
                    images = [img_a, img_b, img_c]
                    div_loss = diversity_loss(images)
                    sim_loss, best_img = similarity_loss(original_image, images)
                    total_loss = div_loss + 0.3*sim_loss
                    print("div_loss",div_loss)
                    print("sim_loss",sim_loss)

                    if total_loss < min_total_loss:
                        min_total_loss = total_loss
                        best_combination = (img_a, img_b, img_c, best_img)
                        # Get the path corresponding to best_img
                        if best_img is not None:
                            best_img_str = tensor_to_str(best_img)
                            if best_img_str in [tensor_to_str(img) for img in cf_candidates_A]:
                                best_cf_path = os.path.join(folder_A, filenames[[tensor_to_str(img) for img in cf_candidates_A].index(best_img_str)])
                            elif best_img_str in [tensor_to_str(img) for img in cf_candidates_B]:
                                best_cf_path = os.path.join(folder_B, filenames[[tensor_to_str(img) for img in cf_candidates_B].index(best_img_str)])
                            elif best_img_str in [tensor_to_str(img) for img in cf_candidates_C]:
                                best_cf_path = os.path.join(folder_C, filenames[[tensor_to_str(img) for img in cf_candidates_C].index(best_img_str)])
                            
        
        if best_combination is not None:
            # Convert tensors to string identifiers for comparison
            best_combination_str = [tensor_to_str(img) for img in best_combination]

            best_cf_A = os.path.join(folder_A, filenames[[tensor_to_str(img) for img in cf_candidates_A].index(best_combination_str[0])])
            best_cf_B = os.path.join(folder_B, filenames[[tensor_to_str(img) for img in cf_candidates_B].index(best_combination_str[1])])
            best_cf_C = os.path.join(folder_C, filenames[[tensor_to_str(img) for img in cf_candidates_C].index(best_combination_str[2])])

            results.append((f'{i - 1:06d}', best_cf_A, best_cf_B, best_cf_C, best_cf_path))
            print(best_cf_A)
            print(best_cf_B)
            print(best_cf_C)
            print(best_cf_path)

    return results

# Save results to CSV with absolute paths
def save_results(results, output_file):
    df = pd.DataFrame(results, columns=['Folder', 'CF1', 'CF2', 'CF3', 'bestCF'])
    df.to_csv(output_file, index=False)

# Run the process
results = find_best_counterfactuals(num_samples=20)
save_results(results, output_file)

# import os
# import numpy as np
# import pandas as pd
# from PIL import Image
# import torch
# from tqdm import tqdm

# # Define paths
# base_folder = '/root/DiME-main/celebA_dataset/img_align_celeba'
# folder_A_base = '/root/autodl-tmp/output0617_CelebA/x_t/exp/name'
# folder_B_base = '/root/DiME-main/output0617_2CelebA/x_t/exp/name'
# folder_C_base = '/root/DiME-main/output0617_3CelebA/x_t/exp/name'

# # Define output files
# output_files = ['/root/DiME-main/loss/information_cf_beta_0.3.csv',
#                 '/root/DiME-main/loss/information_cf_bets_0.6.csv',
#                 '/root/DiME-main/loss/information_cf_beta_0.9.csv',
#                 '/root/DiME-main/loss/information_cf_beta_1.2.csv',
#                 '/root/DiME-main/loss/information_cf_beta_1.5.csv']

# # Load image and resize to 256x256
# def load_image(image_path):
#     image = Image.open(image_path).convert('RGB')
#     image = image.resize((256, 256))  # Resize to 256x256
#     image_tensor = torch.tensor(np.array(image)).float() / 255.0
#     image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
#     return image_tensor

# # Calculate cosine similarity loss
# def calculate_cosine_similarity(image1, image2):
#     image1_flat = image1.flatten()
#     image2_flat = image2.flatten()
#     similarity = torch.dot(image1_flat, image2_flat) / (torch.norm(image1_flat) * torch.norm(image2_flat))
#     cosine_similarity_loss = 1 - similarity
#     return cosine_similarity_loss.item()

# # Define loss functions
# def diversity_loss(images):
#     total_loss = 0.0
#     K = len(images)
#     for i in range(K):
#         for j in range(i + 1, K):
#             cos_sim = calculate_cosine_similarity(images[i], images[j])
#             total_loss += max(0, 0.05 - cos_sim)  # Adjusted threshold for diversity loss
#     total_loss = 2 / (K * (K - 1)) * total_loss
#     return total_loss

# def similarity_loss(original, images):
#     min_loss = float('inf')
#     best_img = None
#     for img in images:
#         cos_sim = calculate_cosine_similarity(original, img)
#         if cos_sim < min_loss:
#             min_loss = cos_sim
#             best_img = img
#     return min_loss, best_img

# # Convert a tensor to a string identifier
# def tensor_to_str(tensor):
#     return str(tensor.numpy().flatten().tolist())

# def find_best_counterfactuals(num_samples=1000, beta_values=[0.3, 0.6, 0.9, 1.2, 1.5]):
#     all_results = []

#     for beta in beta_values:
#         results = []

#         for i in tqdm(range(1, num_samples + 1), desc=f"Processing samples for beta={beta}"):
#             original_image_path = os.path.join(base_folder, f'06{i:04d}.jpg')
#             original_image = load_image(original_image_path)

#             folder_A = os.path.join(folder_A_base, f'{i - 1:06d}')
#             folder_B = os.path.join(folder_B_base, f'{i - 1:06d}')
#             folder_C = os.path.join(folder_C_base, f'{i - 1:06d}')

#             filenames = ['0035.jpg', '0036.jpg', '0037.jpg', '0038.jpg', '0039.jpg']

#             cf_candidates_A = [load_image(os.path.join(folder_A, fn)) for fn in filenames]
#             cf_candidates_B = [load_image(os.path.join(folder_B, fn)) for fn in filenames]
#             cf_candidates_C = [load_image(os.path.join(folder_C, fn)) for fn in filenames]

#             best_combination = None
#             min_total_loss = float('inf')
#             best_cf_path = None

#             for img_a in cf_candidates_A:
#                 for img_b in cf_candidates_B:
#                     for img_c in cf_candidates_C:
#                         images = [img_a, img_b, img_c]
#                         div_loss = diversity_loss(images)
#                         sim_loss, best_img = similarity_loss(original_image, images)
#                         total_loss = div_loss + beta * sim_loss

#                         if total_loss < min_total_loss:
#                             min_total_loss = total_loss
#                             best_combination = (img_a, img_b, img_c, best_img)
#                             # Get the path corresponding to best_img
#                             if best_img is not None:
#                                 best_img_str = tensor_to_str(best_img)
#                                 if best_img_str in [tensor_to_str(img) for img in cf_candidates_A]:
#                                     best_cf_path = os.path.join(folder_A, filenames[[tensor_to_str(img) for img in cf_candidates_A].index(best_img_str)])
#                                 elif best_img_str in [tensor_to_str(img) for img in cf_candidates_B]:
#                                     best_cf_path = os.path.join(folder_B, filenames[[tensor_to_str(img) for img in cf_candidates_B].index(best_img_str)])
#                                 elif best_img_str in [tensor_to_str(img) for img in cf_candidates_C]:
#                                     best_cf_path = os.path.join(folder_C, filenames[[tensor_to_str(img) for img in cf_candidates_C].index(best_img_str)])

#             if best_combination is not None:
#                 # Convert tensors to string identifiers for comparison
#                 best_combination_str = [tensor_to_str(img) for img in best_combination]

#                 best_cf_A = os.path.join(folder_A, filenames[[tensor_to_str(img) for img in cf_candidates_A].index(best_combination_str[0])])
#                 best_cf_B = os.path.join(folder_B, filenames[[tensor_to_str(img) for img in cf_candidates_B].index(best_combination_str[1])])
#                 best_cf_C = os.path.join(folder_C, filenames[[tensor_to_str(img) for img in cf_candidates_C].index(best_combination_str[2])])

#                 results.append((f'{i - 1:06d}', best_cf_A, best_cf_B, best_cf_C, best_cf_path))

#         all_results.append(results)

#     return all_results

# # Save results to CSV with absolute paths
# def save_results(results_list, output_files):
#     for i, results in enumerate(results_list):
#         df = pd.DataFrame(results, columns=['Folder', 'CF1', 'CF2', 'CF3', 'bestCF'])
#         df.to_csv(output_files[i], index=False)
#         print(f"Results for beta={beta_values[i]} saved to {output_files[i]}")

# # Run the process
# beta_values = [0.3, 0.6, 0.9, 1.2, 1.5]
# results_list = find_best_counterfactuals(num_samples=1000, beta_values=beta_values)
# save_results(results_list, output_files)

# import os
# import numpy as np
# import pandas as pd
# from PIL import Image
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from tqdm import tqdm

# # Define paths
# base_folder = '/root/DiME-main/celebA_dataset/img_align_celeba'
# folder_A_base = '/root/autodl-tmp/output0617_CelebA/x_t/exp/name'
# folder_B_base = '/root/DiME-main/output0617_2CelebA/x_t/exp/name'
# folder_C_base = '/root/DiME-main/output0617_3CelebA/x_t/exp/name'

# # Define output files
# output_files = ['/root/DiME-main/loss/information_cf_beta_0.3.csv',
#                 '/root/DiME-main/loss/information_cf_bets_0.6.csv',
#                 '/root/DiME-main/loss/information_cf_beta_0.9.csv',
#                 '/root/DiME-main/loss/information_cf_beta_1.2.csv',
#                 '/root/DiME-main/loss/information_cf_beta_1.5.csv']

# # Define image transformation
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])

# # Custom Dataset
# class ImageDataset(Dataset):
#     def __init__(self, base_folder, indices):
#         self.base_folder = base_folder
#         self.indices = indices

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         image_path = os.path.join(self.base_folder, f'06{self.indices[idx]:04d}.jpg')
#         image = Image.open(image_path).convert('RGB')
#         image = transform(image)
#         return image

# # Load image and resize to 256x256
# def load_image(image_path):
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image)
#     return image

# # Load images from a folder
# def load_images(folder, filenames):
#     images = []
#     for filename in filenames:
#         image_path = os.path.join(folder, filename)
#         image = Image.open(image_path).convert('RGB')
#         image = transform(image)
#         images.append(image)
#     return torch.stack(images)

# # Calculate cosine similarity loss
# def calculate_cosine_similarity(image1, image2):
#     image1_flat = image1.view(-1)
#     image2_flat = image2.view(-1)
#     similarity = torch.dot(image1_flat, image2_flat) / (torch.norm(image1_flat) * torch.norm(image2_flat))
#     cosine_similarity_loss = 1 - similarity
#     return cosine_similarity_loss

# # Define loss functions
# def diversity_loss(images):
#     images_flat = images.view(images.size(0), -1)
#     similarities = torch.mm(images_flat, images_flat.t())
#     norm_products = torch.mm(images_flat.norm(dim=1, keepdim=True), images_flat.norm(dim=1, keepdim=True).t())
#     cosine_similarities = similarities / norm_products
#     losses = 0.05 - cosine_similarities
#     losses = torch.clamp(losses, min=0)
#     total_loss = losses.sum() * 2 / (images.size(0) * (images.size(0) - 1))
#     return total_loss

# def similarity_loss(original, images):
#     min_loss = float('inf')
#     best_img = None
#     for img in images:
#         cos_sim = calculate_cosine_similarity(original, img)
#         if cos_sim < min_loss:
#             min_loss = cos_sim
#             best_img = img
#     return min_loss, best_img

# # Convert a tensor to a string identifier
# def tensor_to_str(tensor):
#     return str(tensor.numpy().flatten().tolist())

# def find_best_counterfactuals(num_samples=1000, beta_values=[0.3, 0.6, 0.9, 1.2, 1.5]):
#     all_results = []
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     for beta in beta_values:
#         results = []

#         for i in tqdm(range(1, num_samples + 1), desc=f"Processing samples for beta={beta}"):
#             original_image_path = os.path.join(base_folder, f'06{i:04d}.jpg')
#             original_image = load_image(original_image_path).to(device)

#             folder_A = os.path.join(folder_A_base, f'{i - 1:06d}')
#             folder_B = os.path.join(folder_B_base, f'{i - 1:06d}')
#             folder_C = os.path.join(folder_C_base, f'{i - 1:06d}')

#             filenames = ['0035.jpg', '0036.jpg', '0037.jpg', '0038.jpg', '0039.jpg']

#             cf_candidates_A = load_images(folder_A, filenames).to(device)
#             cf_candidates_B = load_images(folder_B, filenames).to(device)
#             cf_candidates_C = load_images(folder_C, filenames).to(device)

#             best_combination = None
#             min_total_loss = float('inf')
#             best_cf_path = None

#             for img_a in cf_candidates_A:
#                 for img_b in cf_candidates_B:
#                     for img_c in cf_candidates_C:
#                         images = torch.stack([img_a, img_b, img_c])
#                         div_loss = diversity_loss(images)
#                         sim_loss, best_img = similarity_loss(original_image, images)
#                         total_loss = div_loss + beta * sim_loss

#                         if total_loss < min_total_loss:
#                             min_total_loss = total_loss
#                             best_combination = (img_a, img_b, img_c, best_img)
#                             # Get the path corresponding to best_img
#                             if best_img is not None:
#                                 best_img_str = tensor_to_str(best_img)
#                                 if best_img_str in [tensor_to_str(img) for img in cf_candidates_A]:
#                                     best_cf_path = os.path.join(folder_A, filenames[[tensor_to_str(img) for img in cf_candidates_A].index(best_img_str)])
#                                 elif best_img_str in [tensor_to_str(img) for img in cf_candidates_B]:
#                                     best_cf_path = os.path.join(folder_B, filenames[[tensor_to_str(img) for img in cf_candidates_B].index(best_img_str)])
#                                 elif best_img_str in [tensor_to_str(img) for img in cf_candidates_C]:
#                                     best_cf_path = os.path.join(folder_C, filenames[[tensor_to_str(img) for img in cf_candidates_C].index(best_img_str)])

#             if best_combination is not None:
#                 # Convert tensors to string identifiers for comparison
#                 best_combination_str = [tensor_to_str(img) for img in best_combination]

#                 best_cf_A = os.path.join(folder_A, filenames[[tensor_to_str(img) for img in cf_candidates_A].index(best_combination_str[0])])
#                 best_cf_B = os.path.join(folder_B, filenames[[tensor_to_str(img) for img in cf_candidates_B].index(best_combination_str[1])])
#                 best_cf_C = os.path.join(folder_C, filenames[[tensor_to_str(img) for img in cf_candidates_C].index(best_combination_str[2])])

#                 results.append((f'{i - 1:06d}', best_cf_A, best_cf_B, best_cf_C, best_cf_path))

#         all_results.append(results)

#     return all_results

# # Save results to CSV with absolute paths
# def save_results(results_list, output_files):
#     for i, results in enumerate(results_list):
#         df = pd.DataFrame(results, columns=['Folder', 'CF1', 'CF2', 'CF3', 'bestCF'])
#         df.to_csv(output_files[i], index=False)
#         print(f"Results for beta={beta_values[i]} saved to {output_files[i]}")

# # Run the process
# beta_values = [0.3, 0.6, 0.9, 1.2, 1.5]
# results_list = find_best_counterfactuals(num_samples=1000, beta_values=beta_values)
# save_results(results_list, output_files)

