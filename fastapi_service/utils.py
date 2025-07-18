import io
from PIL import Image
import asyncio
from fastapi import UploadFile

def decode_to_rgb(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")

async def uploadfile_to_rgb_image(file: UploadFile) -> Image.Image:
    data = await file.read()                 # async I/O
    img = await asyncio.to_thread(decode_to_rgb, data)  # run sync decode in thread
    return img
