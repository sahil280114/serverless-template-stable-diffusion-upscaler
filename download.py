# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from diffusers import StableDiffusionUpscalePipeline
import torch

def download_model():
    # do a dry run of loading the huggingface model, which will download weights at build time
    #Set auth token which is required to download stable diffusion model weights
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    model = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)

if __name__ == "__main__":
    download_model()