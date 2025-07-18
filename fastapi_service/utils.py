import io
from PIL import Image
import asyncio
from fastapi import UploadFile

def decode_to_rgb(data: bytes) -> Image.Image:
    """Decode raw image bytes into a PIL Image in RGB mode."""
    return Image.open(io.BytesIO(data)).convert("RGB")

async def uploadfile_to_rgb_image(file: UploadFile) -> Image.Image:
    """Read the uploaded file asynchronously, then convert to RGB Image using a background thread for the blocking decode operation."""
    data = await file.read()
    img = await asyncio.to_thread(decode_to_rgb,data)
    return img
