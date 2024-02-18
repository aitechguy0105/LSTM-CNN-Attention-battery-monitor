import requests
import json

url = "https://stablediffusionapi.com/api/v5/controlnet"

payload = json.dumps({
  "key": "hnC2fAWWr2RidMZfvwzQRaEADsVbnpudoqOhHgfMhX2Z8ZB9jq3RWfqFCn4a",
  "controlnet_model": "canny",
  "controlnet_type": "canny",
  "model_id": "midjourney",
  "auto_hint": "yes",
  "guess_mode": "no",
  "prompt": "make the wonderful interior image in modern style from this sketch, photorealistic image, emphasizing natural light coming through the windows, adding realistic textures to the sofa and wooden floor, and enhancing the overall look to make it appear as a real, lived-in space, bright color, stylish design, ultra high resolution, 4K image",
  "negative_prompt": "deformed frames, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid",
  "init_image": "https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/converted/control_human_openpose.png",
  "mask_image": None,
  "width": "512",
  "height": "512",
  "samples": "1",
  "scheduler": "UniPCMultistepScheduler",
  "num_inference_steps": "30",
  "safety_checker": "no",
  "enhance_prompt": "yes",
  "guidance_scale": 7.5,
  "strength": 0.55,
  "lora_model": None,
  "tomesd": "yes",
  "use_karras_sigmas": "yes",
  "vae": None,
  "lora_strength": None,
  "embeddings_model": None,
  "seed": None,
  "webhook": None,
  "track_id": None
})

headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)