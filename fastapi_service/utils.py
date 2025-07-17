import io
from PIL import Image
import asyncio
from fastapi import UploadFile
import pickle
import os

def decode_to_rgb(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")

async def uploadfile_to_rgb_image(file: UploadFile) -> Image.Image:
    data = await file.read()                 # async I/O
    img = await asyncio.to_thread(decode_to_rgb, data)  # run sync decode in thread
    return img

def load_pickle(path):
    with open(path, 'rb') as f:
        loaded = pickle.load(f)
    print(f"Model + Labels loaded from {path}")
    return loaded

def save_pickle(path, payload):
    os.makedirs(path, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model + Labels saved to {path}")
