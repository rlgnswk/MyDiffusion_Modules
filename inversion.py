from typing import Optional, Union, Tuple, List, Callable, Dict
#from tqdm.notebook import tqdm
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image

from datetime import datetime
import os 
from null_inv_module import *
from p2p_module import *
from utils import *
import json

import argparse
parser = argparse.ArgumentParser(description="Stable Diffusion Inversion Script")
parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
parser.add_argument('--prompt', type=str, required=True, help="Text prompt for the inversion process.")
parser.add_argument('--out_dir', type=str, default="./output", help="Directory to save the output images.")
parser.add_argument('--model_type', type=str, default="CompVis/stable-diffusion-v1-4", help="Directory to save the output images.")
parser.add_argument('--guidance_scale', type=int, default=7.5, help="guidance_scale.")
parser.add_argument('--num_ddim_steps', type=int, default=50, help="guidance_scale.")
parser.add_argument("--isCfgInv", action="store_true", default=False, help="If set, use CFG for DDIM Inversion (default: False)")
parser.add_argument("--isCfgfor", action="store_true", default=False, help="If set, use forward pass with CFG (default: False)")
args = parser.parse_args()

# python inversion.py --image_path "/prompt-to-prompt/example_images/gnochi_mirror.jpeg" --prompt "a cat sitting next to a mirror" --out_dir "output_example_cat"

if __name__ == "__main__":
    #폴더 생성 
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"{args.out_dir}_{current_time_str}"
    os.makedirs(out_dir, exist_ok=True)
    # args 설정 저장
    args_dict = vars(args)  # argparse.Namespace를 딕셔너리로 변환
    with open(os.path.join(out_dir, "args_config.json"), "w") as f:
        json.dump(args_dict, f, indent=4)  # JSON 파일로 저장
        
    #디바이스 설정 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    #print("###############################", device, "###############################")
    print("torch.__version__: ", torch.__version__)        
    print("torch.version.cuda: ", torch.version.cuda)      
    print("torch.cuda.is_available(): ", torch.cuda.is_available())
    
    pipe = StableDiffusionPipeline.from_pretrained(args.model_type).to(device)
    image_path = args.image_path
    prompt = args.prompt
    
    num_ddim_steps = args.num_ddim_steps
    scheduler = None
    #scheduler_ddim = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    null_inversion = NullInversion(pipe, scheduler, device, NUM_DDIM_STEPS = num_ddim_steps, GUIDANCE_SCALE = args.guidance_scale, MAX_NUM_WORDS = 77)
    
    # Preprocessing
    
    null_inversion.init_prompt(prompt)
    ptp_utils.register_attention_control(null_inversion.model, None)
    offsets=(0,0,0,0)
    image_gt = load_512(image_path, *offsets)
    
    ## inv 이렇게 해야함
    
    if args.isCfgInv:
        image_enc, ddim_latents = null_inversion.ddim_inversion_cfg(image_gt)
    else:
        image_enc, ddim_latents = null_inversion.ddim_inversion(image_gt)
    
    if args.isCfgfor:
        naive_image = naive_forward_from_latent(ddim_latents[-1], null_inversion.model, prompt, isCfgfor=True)
    else:
        naive_image = naive_forward_from_latent(ddim_latents[-1], null_inversion.model, prompt, isCfgfor=False)
    
    num_inner_steps=10 
    early_stop_epsilon=1e-5
    
    prompts = [args.prompt]
    
    #controller = EmptyControl() #AttentionStore()
    controller = AttentionStore()
    
    uncond_embeddings = null_inversion.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
    
    # null-inv results
    
    image_inv = naive_forward_from_latent(ddim_latents[-1], null_inversion.model, prompt, isCfgfor=True, null_trained_embeddings=uncond_embeddings)
    
    '''
    if args.isCfgInv:
        _, ddim_latents_temp = null_inversion.ddim_inversion(image_gt)
        uncond_embeddings = null_inversion.null_optimization(ddim_latents_temp, num_inner_steps, early_stop_epsilon)
        image_inv, x_t = text2image_ldm_stable(pipe, prompts, controller, latent=ddim_latents_temp[-1], num_inference_steps=num_ddim_steps, guidance_scale=args.guidance_scale, uncond_embeddings=uncond_embeddings)
    else:
        uncond_embeddings = null_inversion.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        image_inv, x_t = text2image_ldm_stable(pipe, prompts, controller, latent=ddim_latents[-1], num_inference_steps=num_ddim_steps, guidance_scale=args.guidance_scale, uncond_embeddings=uncond_embeddings)
    '''
    # image_inv = run_and_display(pipe, prompts, controller, run_baseline=False, latent=ddim_latents[-1], uncond_embeddings=uncond_embeddings, verbose=False)
    
    # null_inversion.null_optimization
    
    # (image_gt, image_enc), x_t, uncond_embeddings, naive_image = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True, isCfgInv=args.isCfgInv, isCfgfor=args.isCfgfor)

    #naive_image = run_and_display(pipe, prompts, controller, run_baseline=False, latent=ddim_latents[-1], verbose=False, isCfgfor=args.isCfgfor)
    
    print("showing from left to right: the ground truth image, the vq-autoencoder reconstruction, the null-text inverted image")
    #results = [image_gt, image_enc, naive_image, image_inv[0]]
    results = [image_gt, image_enc, naive_image, image_inv]
    import os 
    os.makedirs(f"./{out_dir}", exist_ok=True)
    save_concatenated_image(results, f"./{out_dir}/combined_results.png", orientation="horizontal")
    