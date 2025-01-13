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

'''# Stable Diffusion 기본값인듯 
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = '' #null token
LOW_RESOURCE = False  # ?
NUM_DDIM_STEPS = 50 # step 50
GUIDANCE_SCALE = 7.5 # CFG
MAX_NUM_WORDS = 77 # max length
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN, scheduler=scheduler).to(device)
try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")
tokenizer = ldm_stable.tokenizer'''

################################### Null T Code ###################################

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image
#scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
class NullInversion:
    def __init__(self, model, scheduler_ddim, device, NUM_DDIM_STEPS = 50, GUIDANCE_SCALE = 7.5, MAX_NUM_WORDS = 77):
        # Stable Diffusion 기본값인듯 
        #self.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.MY_TOKEN = '' #null token
        self.LOW_RESOURCE = False  # ?
        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS # step 50
        self.GUIDANCE_SCALE = GUIDANCE_SCALE # CFG
        self.MAX_NUM_WORDS = MAX_NUM_WORDS # max length
        self.device = device #torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model#StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=self.MY_TOKEN, scheduler=scheduler).to(device)
        self.model.scheduler.set_timesteps(self.NUM_DDIM_STEPS)
        try:
            self.model.disable_xformers_memory_efficient_attention()
        except AttributeError:
            print("Attribute disable_xformers_memory_efficient_attention() is missing")
        self.tokenizer = self.model.tokenizer
        self.prompt = None
        self.context = None
        
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else self.GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in tqdm(range(self.NUM_DDIM_STEPS)):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @torch.no_grad()
    def ddim_loop_cfg(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in tqdm(range(self.NUM_DDIM_STEPS)):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            #noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            
            noise_pred_uncond = self.get_noise_pred_single(latent, t, uncond_embeddings)
            noise_pred_cond = self.get_noise_pred_single(latent, t, cond_embeddings)
            noise_pred = noise_pred_uncond + self.GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)

            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent
    
    @torch.no_grad()
    def ddim_loop_cfg_calScale(self, latent):
        """
        '두 번째 코드(invert)'의 DDIM Inversion 방식을
        최대한 모사하여, cfg + scale_model_input 로직을 적용한 버전.
        
        Args:
            latent: VAE.encode(...)로부터 얻은 초기 latent (x_T)
        Returns:
            all_latent: 각 스텝에서의 latent를 저장한 리스트
        """
        device = self.model.device
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        
        # -----------------------------
        #  1) DDIM 스케줄러 세팅
        # -----------------------------
        
        self.model.scheduler.set_timesteps(self.NUM_DDIM_STEPS, device=device)
        # timesteps를 역순으로
        
        timesteps = list(reversed(self.model.scheduler.timesteps))  # ex) [49,48,...,0]

        all_latent = []
        latents = latent.clone().detach().to(device)
        
        # -----------------------------
        #  2) DDIM 역방향 루프
        # -----------------------------
        for i in tqdm(range(1, self.NUM_DDIM_STEPS), total=self.NUM_DDIM_STEPS - 1):
            # (두 번째 코드처럼) 마지막 스텝은 스킵
            if i >= self.NUM_DDIM_STEPS - 1:
                continue
            
            t = timesteps[i]  # 역순 timesteps에서 i번째
            
            # (A) scale_model_input
            #     - CFG를 위해 latents를 2배로 확장 후 스케일링
            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)

            # (B) UNet forward
            # uncond_embeddings와 cond_embeddings를 붙여서 (B, seqlen, dim) 형태
            context = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
            noise_pred = self.model.unet(
                latent_model_input,
                t,
                encoder_hidden_states=context
            )["sample"]

            # (C) CFG
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)

            # (D) DDIM Inversion 수식 (두 번째 코드와 동일)
            # current_t = t - (1000//NUM_DDIM_STEPS), next_t = t
            current_t = max(0, t.item() - (1000 // self.NUM_DDIM_STEPS))
            next_t = t.item()
            alpha_t = self.model.scheduler.alphas_cumprod[current_t]
            alpha_t_next = self.model.scheduler.alphas_cumprod[next_t]

            # latents_{t} = ( latents_{t+1} - sqrt(1-alpha_t)*noise_pred ) * (...)
            latents = (
                (latents - (1 - alpha_t).sqrt() * noise_pred)
                * (alpha_t_next.sqrt() / alpha_t.sqrt())
                + (1 - alpha_t_next).sqrt() * noise_pred
            )

            all_latent.append(latents.clone().detach())

        return all_latent


    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    @torch.no_grad()
    def ddim_inversion_cfg(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop_cfg(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * self.NUM_DDIM_STEPS)
        for i in range(self.NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + self.GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list
    
    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False, isCfgInv=True, isCfgfor=True):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
            
        if isCfgInv:
            image_rec, ddim_latents = self.ddim_inversion_cfg(image_gt)
        else:
            image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        if verbose:
            print("Generating naive image from DDIM latent...")
        
        if isCfgfor:
            naive_image = naive_forward_from_latent(ddim_latents[-1], self.model, prompt, isCfgfor=True)
        else:
            naive_image = naive_forward_from_latent(ddim_latents[-1], self.model, prompt, isCfgfor=False)
            
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings, naive_image
        


################################### Inference Code  ###################################

@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image'
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent


def run_and_display(model ,prompts, controller, num_inference_steps=50, guidance_scale=7.5, latent=None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True, isCfgfor=True):
    #if run_baseline:
    print("w.o. prompt-to-prompt")
    images, latent = run_and_display(model, prompts, controller, latent=latent, run_baseline=False, generator=generator)
    #print("with prompt-to-prompt")
    #images, x_t = text2image_ldm_stable(model, prompts, controller, latent=latent, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator, uncond_embeddings=uncond_embeddings, isCfgfor=isCfgfor)
    if verbose:
        ptp_utils.view_images(images)
    return images#, x_t

@torch.no_grad()
def naive_forward_from_latent(latent, model, prompt, num_inference_steps=50, isCfgfor=True, GUIDANCE_SCALE=7.5, null_trained_embeddings=None):
    model.scheduler.set_timesteps(num_inference_steps-1)
    
    # 조건부 임베딩
    text_input = model.tokenizer(
        [prompt],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    
    # 무조건부(null prompt) 임베딩
    if null_trained_embeddings is None:
        null_input = model.tokenizer(
            [""],  # null prompt
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        null_embeddings = model.text_encoder(null_input.input_ids.to(model.device))[0]

        

    # latent를 GPU로 이동
    latent = latent.to(model.device)
    
    # latent 크기 조정
    if latent.shape[-2:] != (64, 64):
        latent = torch.nn.functional.interpolate(latent, size=(64, 64), mode='nearest')

    # DDIM 스케줄링
    for i, t in enumerate(tqdm(model.scheduler.timesteps)):
        # 조건부와 무조건부를 함께 전달
        latent_input = torch.cat([latent] * 2)
        if null_trained_embeddings is None:
            encoder_hidden_states = torch.cat([null_embeddings, text_embeddings], dim=0)  # 무조건부 + 조건부
        else:
            encoder_hidden_states = torch.cat([null_trained_embeddings[i].expand(*text_embeddings.shape), text_embeddings], dim=0)
            
        noise_pred = model.unet(latent_input, t, encoder_hidden_states=encoder_hidden_states)["sample"]
        
        # 조건부와 무조건부로 분리
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        
        # Classifier-Free Guidance 적용
        if isCfgfor:
            noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond
        
        # latent 업데이트
        latent = model.scheduler.step(noise_pred, t, latent).prev_sample

    # 최종 이미지를 생성
    image = model.vae.decode(1 / 0.18215 * latent)["sample"]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).astype(np.uint8)
    
    return Image.fromarray(image)