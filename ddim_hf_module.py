import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler

def invert(
    start_latents,
    prompt,
    pipe,
    device,
    guidance_scale=3.5,
    num_inference_steps=80,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        print(latents.shape)
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        #print(latents.shape)
        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (1 - alpha_t_next).sqrt() * noise_pred

        # Store
        #intermediate_latents.append(latents)

    return latents

def load_image(image_path, size=None):
    """
    Load an image from a given path and optionally resize it.

    Args:
        image_path (str): Path to the input image.
        size (tuple, optional): Desired size (width, height) of the output image.

    Returns:
        PIL.Image.Image: Loaded image.
    """
    img = Image.open(image_path).convert("RGB")  # Load and convert to RGB
    if size is not None:
        img = img.resize(size)
    return img


if __name__ == "__main__":
    prompt = "a cat sitting next to a mirror"

    GUIDANCE_SCALE = 7.5 # CFG
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("###############################",device, "###############################")
    print(torch.__version__)        
    print(torch.version.cuda)      
    print(torch.cuda.is_available())  

    # Load a pipeline
    #pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32).to(device)
    #pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
    # Set up a DDIM scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    generator = torch.manual_seed(42)
    output_dir = "ddim_gen_inv_seed_compvis"
    import os
    os.makedirs(output_dir, exist_ok=True)

    input_image_prompt = prompt
    # load target image
    input_image_path = f"/shared/s2/lab01/gihoon/prompt-to-prompt/example_images/gnochi_mirror.jpeg"
    input_image = load_image(input_image_path)

    for steps in [1, 10, 25, 50, 100, 200, 500, 999]:
        
        

        
        num_inference_steps = steps  # 원하는 스텝 수
        
        #print(f"Image saved to {output_path}")

        
        
        # Encode with VAE
        with torch.no_grad():
            latent = pipe.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1)
        #l = 0.18215 * latent.latent_dist.sample()
        l = 0.18215 * latent.latent_dist.mean
        with torch.no_grad():
            inverted_latents = invert(l, input_image_prompt, num_inference_steps=num_inference_steps, guidance_scale=GUIDANCE_SCALE)

        print("inverted_latents.shape: ", inverted_latents.shape)

        with torch.no_grad():
            im = pipe.decode_latents(inverted_latents)
        out_image = pipe.numpy_to_pil(im)[0]

        output_path = f"{output_dir}/inverted_noise_{steps}.png"  # File path for saving the image
        out_image.save(output_path)

        print(f"Image saved to {output_path}")

        #prompt = "Beautiful DSLR Photograph of a penguin on the beach, golden hour"
        #negative_prompt = "blurry, ugly, stock photo"
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        im = pipe(prompt, latents=inverted_latents, generator=generator, num_inference_steps=steps, guidance_scale=GUIDANCE_SCALE).images[0]
        im.resize((256, 256))  # Resize for convenient viewing

        output_path = f"{output_dir}/inverted_image_{steps}.png"  # File path for saving the image
        im.save(output_path)
        
        print(f"Image saved to {output_path}")