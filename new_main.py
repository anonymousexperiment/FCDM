#%%
import os
import yaml
import math
import random
import argparse
import itertools
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from PIL import Image
from time import time
from os import path as osp
from multiprocessing import Pool

import torch
from torch.utils import data

from torchvision import transforms
from torchvision import datasets

from core import dist_util
from core.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_classifier,
    args_to_dict,
    add_dict_to_argparser,
)
from core.sample_utils import (
    get_DiME_iterative_sampling,
    clean_class_cond_fn,
    dist_cond_fn,
    ImageSaver,
    SlowSingleLabel,
    Normalizer,
    load_from_DDP_model,
    PerceptualLoss,
    X_T_Saver,
    Z_T_Saver,
    ChunkedDataset,
)
from core.image_datasets import CelebADataset, CelebAMiniVal, CelebAHQDataset,UTKFaceDataSet
from core.gaussian_diffusion import _extract_into_tensor
from core.classifier.densenet import ClassificationModel

import matplotlib
matplotlib.use('Agg')  # to disable display
import tqdm
import csv

# =======================================================
# =======================================================
# Functions
# =======================================================
# =======================================================

def create_args():
    defaults = dict(
        clip_denoised = True,
        batch_size = 6,
        gpu='0',
        num_batches=20000,
        use_train=False,
        dataset='CelebAHQ',#,CelebAHQ,UTKFace

        # path args celebA
        # output_path=r'/root/autodl-tmp/output0627_CelebA_male_young',
        # classifier_path=r'./models/classifier.pth',
        # oracle_path=r'./models/oracle.pth',
        # model_path=r"./models/ddpm-celeba.pt",
        # data_dir=r"./celebA_dataset",#celebA_dataset,celebAHQ_dataset,UTKFace_dataset
        # exp_name='',

        #celebAHQ
        output_path=r'/root/autodl-tmp/output0726_CelebAGQ_M_W',
        # classifier_path=r'./models/classifier_celebAHQ.pth',
        classifier_path=r'./models/classifier.pth',
        oracle_path=r'./models/oracle.pth',
        model_path=r"./models/ddpm-celeba.pt",
        data_dir=r"/root/autodl-tmp/celebAHQ/CelebA-HQ-256",#celebA_dataset,celebAHQ_dataset,UTKFace_dataset
        exp_name='',


        # sampling args
        #classifier_scales=[8,10,15],
        classifier_scales=[12],
        seed=4,
        query_label=[21,36],    
        weight_original=[1,1],#celebA
        #weight_original=[1,1,0.338,0.662],
    
        target_label=1,
        use_ddim=True,
        start_step=40,
        timestep_respacing=80,
        use_logits=False,
        l1_loss=0.0,
        l2_loss=0.0,
        l_perc=0.0,
        l_perc_layer=1,
        use_sampling_on_x_t=True,
        sampling_scale=1.,  # use this flag to rescale the variance of the noise
        guided_iterations=9999999,  # set a high number to do all iteration in a guided way

        # evaluation args
        merge_and_eval=False,  # when all chunks have finished, run it with this flag

        # misc args
        num_chunks=1,
        chunk=0,
        save_x_t=True,
        save_z_t=True,
        save_images=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser.parse_args()

args=create_args()
os.makedirs(args.output_path,exist_ok=True)

#%%#CelebADataset,CelebAHQDataset,UTKFaceDataSet
dataset = CelebAHQDataset(image_size=args.image_size,
                        data_dir=args.data_dir,
                        partition='train',
                        random_crop=False,
                        random_flip=False,
                        query_label=args.query_label,
                        start_index=0
                        )
#print("dataset.size!!!!!!!!!!!",dataset.data.size)

if len(dataset) - args.batch_size * args.num_batches > 0:
    dataset = SlowSingleLabel( dataset=dataset,
                                maxlen=args.batch_size * args.num_batches)

loader = data.DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=0, pin_memory=False)


# ========================================
# load models
print('Loading Model and diffusion model')
model, diffusion = create_model_and_diffusion(
    multiclass=True,
    num_classes=len(args.query_label),
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)

saved_state_dict=dist_util.load_state_dict(args.model_path, map_location="cpu")
for param_name,param in model.named_parameters():
    if param_name in ["label_emb.weight", "label_emb.bias"]:
        saved_state_dict[param_name]=param
model.load_state_dict(
    saved_state_dict
)
model.to(dist_util.dev())
if args.use_fp16:
    model.convert_to_fp16()
model.eval()

def model_fn(x, t, y=None):
    assert y is not None
    return model(x, t, y if args.class_cond else None)

print('Loading Classifier')

classifier = ClassificationModel(args.classifier_path, args.query_label).to(dist_util.dev())
classifier.eval()

# ========================================
# get custom function for the forward phase
# and other variables of interest

sample_fn = get_DiME_iterative_sampling(use_sampling=args.use_sampling_on_x_t)

x_t_saver = X_T_Saver(args.output_path, args.exp_name) if args.save_x_t else None
z_t_saver = Z_T_Saver(args.output_path, args.exp_name) if args.save_z_t else None
save_imgs = ImageSaver(args.output_path, args.exp_name, extention='.jpg') if args.save_images else None

current_idx = 0
start_time = time()

stats = {
    'n': 0,
    'flipped': 0,
    'bkl': [],
    'l_1': [],
    'pred': [],
    'cf pred': [],
    'target': [],
    'label': [],
}

acc = 0
n = 0
classifier_scales = [float(x) for x in args.classifier_scales]

# if os.path.exists("model.ckpt"):
#     model.load_state_dict(torch.load("model.ckpt"))

print('Starting Image Generation')




# with open('data.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     
#     writer.writerow(['Name', 'Age', 'Country'])
#     writer.writerow(['1', '2', '3'])



with open('images_output0726_CelebAJQ_Me_W.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for idx, (img, lab, image_files) in enumerate(loader):
        #for image_file in image_files:
            #print("image_file",image_file)
        lt = list(image_files)
        writer.writerow(lt) 
        print(image_files)
        print(f'[Chunk {args.chunk + 1} / {args.num_chunks}] {idx} / {min(args.num_batches, len(loader))} | Time: {int(time() - start_time)}s')

        img = img.to(dist_util.dev())
        I = (img / 2) + 0.5
        lab = lab.to(dist_util.dev(), dtype=torch.long)
        t = torch.zeros(img.size(0), device=dist_util.dev(),
                        dtype=torch.long)

        # Initial Classification, no noise included
        with torch.no_grad():
            logits = classifier(img)
            pred = (logits > 0).long()  # should be matrix

        acc += (pred == lab).sum().item()
        n += lab.size(0)

        # as the model is binary, the target will always be the inverse of the prediction
        target = torch.ones_like(pred,dtype=pred.dtype,device=pred.device) - pred

        t = torch.ones_like(t) * args.start_step

        # add noise to the input image 
        noise_img = diffusion.q_sample(img, t)

        batch_size,num_targets=lab.size()
        transformed = torch.zeros(size=(batch_size,),dtype=torch.bool,device=lab.device)

        for jdx, classifier_scale in enumerate(tqdm.tqdm(classifier_scales)):
            # choose the target label
            model_kwargs = {}
            model_kwargs['y'] = target[~transformed,:]
            #print("111111",target[~transformed])
            # sample image from the noisy_img
            cfs, xs_t_s, zs_t_s = sample_fn(
                diffusion,
                model_fn,
                img[~transformed, ...].shape,
                args.start_step,
                img[~transformed, ...],
                t,
                z_t=noise_img[~transformed, ...],
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                class_grad_fn=clean_class_cond_fn,
                class_grad_kwargs={'y': target[~transformed,:],
                                    'classifier': classifier,
                                    's': classifier_scale,
                                    'use_logits': args.use_logits},
                dist_grad_fn=dist_cond_fn,
                dist_grad_kargs={'l1_loss': args.l1_loss,
                                    'l2_loss': args.l2_loss,
                                    'l_perc': None},
                guided_iterations=args.guided_iterations,
                is_x_t_sampling=False
            )

            #TODO

            # evaluate the cf and check whether the model flipped the prediction
            with torch.no_grad():
                cfsl = classifier(cfs)
                cfsp = cfsl > 0
            
            if jdx == 0:
                cf = cfs.clone().detach()
                x_t_s = [xp.clone().detach() for xp in xs_t_s]
                z_t_s = [zp.clone().detach() for zp in zs_t_s]

            cf[~transformed] = cfs
            for kdx in range(len(x_t_s)):

                x_t_s[kdx][~transformed] = xs_t_s[kdx]
                z_t_s[kdx][~transformed] = zs_t_s[kdx]
            #weight=torch.tensor([1,0.212,0.788,1,0.338,0.662])
            #weight=torch.tensor([1,1]) 
           
            weight = weight / weight.sum(dim=0,keepdim=True)
            #weight=torch.tensor([1,1,0.338,0.662])
            correctness=(target[~transformed] == cfsp).to(torch.float)
            score=correctness*(weight.to(correctness.device))
            score=torch.sum(score,dim=1)
            transformed[~transformed]=score>0.5 #celebA1.2
            

            
            if transformed.float().sum().item() == transformed.size(0):
                break

        if args.save_x_t:
            x_t_saver(x_t_s)

        if args.save_z_t:
            z_t_saver(z_t_s)

        with torch.no_grad():
            logits_cf = classifier(cf)
            pred_cf = (logits_cf > 0).long() 

            # process images
            cf = ((cf + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            cf = cf.permute(0, 2, 3, 1)
            cf = cf.contiguous().cpu()

            I = (I * 255).to(torch.uint8)
            I = I.permute(0, 2, 3, 1)
            I = I.contiguous().cpu()

            noise_img = ((noise_img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            noise_img = noise_img.permute(0, 2, 3, 1)
            noise_img = noise_img.contiguous().cpu()

            # add metrics
            dist_cf = torch.sigmoid(logits_cf)
            dist_cf[target == 0] = 1 - dist_cf[target == 0]
            bkl = (1 - dist_cf).detach().cpu()

            # dists
            I_f = (I.to(dtype=torch.float) / 255).view(I.size(0), -1)
            cf_f = (cf.to(dtype=torch.float) / 255).view(I.size(0), -1)
            l_1 = (I_f - cf_f).abs().mean(dim=1).detach().cpu()

            stats['l_1'].append(l_1)
            stats['n'] += I.size(0)
            stats['bkl'].append(bkl)
            stats['flipped'] += (pred_cf == target).sum().item()
            stats['cf pred'].append(pred_cf.detach().cpu())
            stats['target'].append(target.detach().cpu())
            stats['label'].append(lab.detach().cpu())
            stats['pred'].append(pred.detach().cpu())
        if args.save_images:
            save_imgs(I.numpy(), cf.numpy(), noise_img.numpy(),
                        target, lab, pred, pred_cf,
                        bkl.numpy(),l_1)
            print(image_files)

        if (idx + 1) == min(args.num_batches, len(loader)):
            print(f'[Chunk {args.chunk + 1} / {args.num_chunks}] {idx + 1} / {min(args.num_batches, len(loader))} | Time: {int(time() - start_time)}s')
            print('\nDone')
            break

        current_idx += I.size(0)

 #       torch.save(model.state_dict(),"model.ckpt")

def mean(array):
    m = np.mean(array).item()
    return 0 if math.isnan(m) else m

