################################ CFDM IS #################################
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
from scipy.stats import entropy
import argparse
 
# python IS.py --input_image_dir ./input_images
default=[
 '/root/autodl-tmp/output0709_CelebAHQ_Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/Results/exp/name/CC/CCF/CF',
    '/root/autodl-tmp/output0710_CelebAHQ_Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/Results/exp/name/CC/CCF/CF'
]
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('--input_image_dir', type=str, default='/root/autodl-tmp/Vikram_et_al./Young_prot_Male')
parser.add_argument('--input_image_dirs', nargs='+', type=str)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--device', type=str, choices=["cuda:0", "cpu"], default="cuda:0")
args = parser.parse_args()
 
 
# we should use same mean and std for inception v3 model in training and testing process
# reference web page: https://pytorch.org/hub/pytorch_vision_inception_v3/
mean_inception = [0.485, 0.456, 0.406]
std_inception = [0.229, 0.224, 0.225]
 
 
def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]
 
def readDir():
    files = []
    for directory in default:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith(".jpg"):  # 
                    files.append(os.path.join(directory, file))
        else:
            print(f"Directory {directory} does not exist.")
    return files
def inception_score(batch_size=args.batch_size, resize=True, splits=10):
    # Set up dtype
    device = torch.device(args.device)  # you can change the index of cuda
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)
 
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()
 
    # Get predictions using pre-trained inception_v3 model
    print('Computing predictions using inception v3 model')
 
    files = readDir()
    #files = read_dirs(args.input_image_dirs)
    N = len(files)
    preds = np.zeros((N, 1000))
    if batch_size > N:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
 
    for i in tqdm(range(0, N, batch_size)):
        start = i
        end = i + batch_size
        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])
 
        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255
 
        batch = torch.from_numpy(images).type(torch.FloatTensor)
        batch = batch.to(device)
        y = get_pred(batch)
        print(y.shape)
        preds[i:i + batch_size] = get_pred(batch)
 
    assert batch_size > 0
    assert N > batch_size
 
    # Now compute the mean KL Divergence
    print('Computing KL Divergence')
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]  # split the whole data into several parts
        py = np.mean(part, axis=0)  # marginal probability
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]  # conditional probability
            scores.append(entropy(pyx, py))  # compute divergence
        split_scores.append(np.exp(scores))
 
    return np.max(split_scores), np.mean(split_scores)
 
 
def read_dirs(dirs):
    all_files = []
    for dir_path in dirs:
        if os.path.isdir(dir_path):
            for root, _, files in os.walk(dir_path):
                for file in files:
                    all_files.append(os.path.join(root, file))
        else:
            print(f"Error: {dir_path} is not a directory")
    return all_files
 
 
if __name__ == '__main__':
 
    MAX, IS = inception_score(splits=10)
    print('MAX IS is %.4f' % MAX)
    print('The average IS is %.4f' % IS)
 


# #################################  other methods IS  #################################
# import os
# import argparse
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torchvision.models import inception_v3
# from PIL import Image
# from scipy.stats import entropy
# from tqdm import tqdm

# # Argument parser
# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--input_image_dir', type=str, default='/root/autodl-tmp/attGAN/celebAHQ/Mouth_Slightly_Open_Wearing_Lipstick_pre_Blond_Hair/Wearing_Lipstick')
# parser.add_argument('--batch_size', type=int, default=16)
# parser.add_argument('--device', type=str, choices=["cuda:0", "cpu"], default="cuda:0")
# parser.add_argument('--num_images', type=int, default=17000, help='Number of images to use for IS calculation')
# args = parser.parse_args()

# # Mean and standard deviation for inception model
# mean_inception = [0.485, 0.456, 0.406]
# std_inception = [0.229, 0.224, 0.225]

# def imread(filename):
#     """
#     Loads an image file into a (height, width, 3) uint8 ndarray.
#     """
#     return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]

# def inception_score(batch_size=args.batch_size, resize=True, splits=10):
#     # Set up dtype
#     device = torch.device(args.device)
#     # Load inception model
#     inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
#     inception_model.eval()
#     up = torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)

#     def get_pred(x):
#         if resize:
#             x = up(x)
#         x = inception_model(x)
#         return F.softmax(x, dim=1).data.cpu().numpy()

#     # Get predictions using pre-trained inception_v3 model
#     print('Computing predictions using inception v3 model')

#     files = read_dirs([args.input_image_dir], args.num_images)
#     N = len(files)
#     preds = np.zeros((N, 1000))
#     if batch_size > N:
#         print(('Warning: batch size is bigger than the data size. '
#                'Setting batch size to data size'))

#     for i in tqdm(range(0, N, batch_size)):
#         start = i
#         end = i + batch_size
#         images = np.array([imread(str(f)).astype(np.float32)
#                            for f in files[start:end]])

#         # Reshape to (n_images, 3, height, width)
#         images = images.transpose((0, 3, 1, 2))
#         images /= 255

#         batch = torch.from_numpy(images).type(torch.FloatTensor)
#         batch = batch.to(device)
#         y = get_pred(batch)
#         preds[i:i + batch_size] = get_pred(batch)

#     assert batch_size > 0
#     assert N > batch_size

#     # Now compute the mean KL Divergence
#     print('Computing KL Divergence')
#     split_scores = []
#     for k in range(splits):
#         part = preds[k * (N // splits): (k + 1) * (N // splits), :]  # split the whole data into several parts
#         py = np.mean(part, axis=0)  # marginal probability
#         scores = []
#         for i in range(part.shape[0]):
#             pyx = part[i, :]  # conditional probability
#             scores.append(entropy(pyx, py))  # compute divergence
#         split_scores.append(np.exp(scores))

#     return np.max(split_scores), np.mean(split_scores)

# def read_dirs(dirs, num_images):
#     all_files = []
#     for dir_path in dirs:
#         if os.path.isdir(dir_path):
#             for root, _, files in os.walk(dir_path):
#                 for file in files:
#                     if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#                         all_files.append(os.path.join(root, file))
#                     if len(all_files) >= num_images:
#                         break
#                 if len(all_files) >= num_images:
#                     break
#         else:
#             print(f"Error: {dir_path} is not a directory")
#     return all_files[:num_images]

# if __name__ == '__main__':
#     MAX, IS = inception_score(splits=10)
#     print('MAX IS is %.4f' % MAX)
#     print('The average IS is %.4f' % IS)



# #################################  Ablation_experiment IS  #################################

# import os
# import numpy as np
# import pandas as pd
# from PIL import Image
# import torch
# from torchvision.models import inception_v3
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# from scipy.stats import entropy
# from tqdm import tqdm
# import argparse

# # Argument parser
# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--csv_file', type=str, default='/root/CFDM/Ablation_experiment/recon_loss_cf.csv')
# parser.add_argument('--batch_size', type=int, default=16)
# parser.add_argument('--device', type=str, choices=["cuda:0", "cpu"], default="cuda:0")
# parser.add_argument('--num_images', type=int, default=25000, help='Number of images to use for IS calculation')
# args = parser.parse_args()

# # Define paths
# csv_file = args.csv_file

# # #################################  Original Method #################################
# # Define the transform
# transform = transforms.Compose([
#     transforms.Resize((299, 299)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Function to calculate inception score
# def calculate_inception_score(images, model, splits=10):
#     N = len(images)
#     assert N > 0

#     preds = np.zeros((N, 1000))

#     for i, img in enumerate(tqdm(images, desc="Calculating Inception Score")):
#         img = transform(img).unsqueeze(0).to(args.device)
#         with torch.no_grad():
#             pred = model(img)
#             preds[i] = F.softmax(pred, dim=1).cpu().numpy()

#     split_scores = []

#     for k in range(splits):
#         part = preds[k * (N // splits): (k + 1) * (N // splits), :]
#         py = np.mean(part, axis=0)
#         scores = [entropy(pyx, py) for pyx in part]
#         split_scores.append(np.exp(np.mean(scores)))

#     return np.mean(split_scores), np.std(split_scores)

# # Function to load images and compute IS
# def compute_is_from_csv(csv_file, model):
#     df = pd.read_csv(csv_file)
#     image_paths = df['bestCF'].tolist()

#     images = []
#     for path in tqdm(image_paths, desc="Loading images"):
#         try:
#             image = Image.open(path).convert('RGB')
#             images.append(image)
#         except Exception as e:
#             print(f"Error loading image {path}: {e}")

#     mean_score, std_score = calculate_inception_score(images, model)
#     return mean_score, std_score

# # #################################  Other Methods IS  #################################
# # Function to read images from a list of paths
# def read_images_from_paths(paths, num_images):
#     images = []
#     for path in paths[:num_images]:
#         try:
#             image = Image.open(path).convert('RGB')
#             images.append(image)
#         except Exception as e:
#             print(f"Error loading image {path}: {e}")
#     return images

# # Function to compute inception score using batch processing
# def inception_score(image_paths, batch_size=args.batch_size, resize=True, splits=10):
#     device = torch.device(args.device)
#     inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
#     inception_model.eval()
#     up = torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)

#     def get_pred(x):
#         if resize:
#             x = up(x)
#         x = inception_model(x)
#         return F.softmax(x, dim=1).data.cpu().numpy()

#     images = read_images_from_paths(image_paths, args.num_images)
#     N = len(images)
#     preds = np.zeros((N, 1000))
#     if batch_size > N:
#         print(('Warning: batch size is bigger than the data size. '
#                'Setting batch size to data size'))

#     for i in tqdm(range(0, N, batch_size)):
#         start = i
#         end = i + batch_size
#         batch_images = images[start:end]

#         batch = torch.stack([transform(img) for img in batch_images]).to(device)
#         preds[start:end] = get_pred(batch)

#     split_scores = []
#     for k in range(splits):
#         part = preds[k * (N // splits): (k + 1) * (N // splits), :]
#         py = np.mean(part, axis=0)
#         scores = [entropy(pyx, py) for pyx in part]
#         split_scores.append(np.exp(np.mean(scores)))

#     return np.mean(split_scores), np.std(split_scores)

# # Main function to calculate and save IS using the chosen method
# def main():
#     device = torch.device(args.device)
#     model = inception_v3(pretrained=True, transform_input=False).to(device)
#     model.eval()

#     mean_score, std_score = compute_is_from_csv(csv_file, model)
#     print(f"Original Method IS: {mean_score} ± {std_score}")

#     df = pd.read_csv(csv_file)
#     image_paths = df['bestCF'].tolist()

#     max_is, avg_is = inception_score(image_paths, splits=10)
#     print(f"Other Method Max IS: {max_is}")
#     print(f"Other Method Avg IS: {avg_is}")

# if __name__ == '__main__':
#     main()


