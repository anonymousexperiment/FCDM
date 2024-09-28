# **Data preparation**
Please download and uncompress the CelebA dataset [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), and the CelebAHQ dataset [here](https://github.com/switchablenorms/CelebAMask-HQ).
. 

python -W ignore main.py \
    --attention_resolutions 32,16,8 \
    --class_cond False \
    --diffusion_steps 500 \
    --learn_sigma True \
    --noise_schedule linear \
    --num_channels 128 \
    --num_heads 4 \
    --num_res_blocks 2 \
    --resblock_updown True \
    --use_fp16 True \
    --use_scale_shift_norm True \
    --batch_size 192 \
    --timestep_respacing 200  \
    --dataset CelebA \
    --exp_name exp/name \
    --gpu 0 \
    --seed 4 \
    --l1_loss 0.05 \
    --use_logits True \
    --l_perc 30 \
    --l_perc_layer 18 \
    --save_x_t True \
    --save_z_t True\
    --use_sampling_on_x_t True \
    --save_images True \
    --image_size 128
