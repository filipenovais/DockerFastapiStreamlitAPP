from fastapi import FastAPI, UploadFile
from model_inference import infer_model
from model_manager import load_model
from utils import uploadfile_to_rgb_image
import asyncio

app = FastAPI()
model = None
labels = None

@app.get("/load_model")
def _load_model(model_path: str):
    global model
    global labels
    model, labels = load_model(model_path+".pkl")
    return {"status": "loaded"}

@app.post("/infer_model")
async def _infer_model(file: UploadFile, top_k: int):
    image = await uploadfile_to_rgb_image(file)
    top_classes = await asyncio.to_thread(infer_model, model, labels, image, top_k)
    return {"top_classes": top_classes}
