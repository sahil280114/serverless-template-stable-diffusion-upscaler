import torch
from diffusers import StableDiffusionUpscalePipeline
import base64
from io import BytesIO
from PIL import Image
import requests

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    # load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    model = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to("cuda")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    image_url = model_inputs.get('imageURL', None)
    if image_url == None:
        return {'message': "No image url provided"}
    
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    

    response = requests.get(image_url)
    low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    low_res_img = low_res_img.resize((128, 128))

    upscaled_image = model(prompt=prompt, image=low_res_img).images[0]
    
    buffered = BytesIO()
    upscaled_image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}
