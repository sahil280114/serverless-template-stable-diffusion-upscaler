# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64
from io import BytesIO
from PIL import Image
import banana_dev as banana
model_inputs = {'prompt': 'a white cat','image_url':'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png'}

#call on a local running gpu
res = requests.post('http://localhost:8000/', json = model_inputs)

image_byte_string = res.json()["image_base64"]

#call on banana
out = banana.run("api_key","model_key",model_inputs)
image_byte_string = out["modelOutputs"][0]["image_base64"]
image_encoded = image_byte_string.encode('utf-8')
image_bytes = BytesIO(base64.b64decode(image_encoded))
image = Image.open(image_bytes)
image.save("output.jpg")