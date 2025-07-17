# app.py
from fastapi import FastAPI, UploadFile, Request, HTTPException, Query
import asyncio

from model_inference import infer_model
from model_manager import load_model
from utils import uploadfile_to_rgb_image

app = FastAPI()

# Initialize runtime state containers on the app instance.
# models_dict caches all loaded models by key (model_path).
# model / labels hold the "active" model last requested via /load_model.
app.state.models_dict = {}
app.state.model = None
app.state.labels = None


@app.get("/load_model")
def load_model_endpoint(model_path: str, request: Request):
    state = request.app.state

    if model_path not in state.models_dict:
        model, labels = load_model(f"{model_path}.pkl")
        state.models_dict[model_path] = (model, labels)
    else:
        state.model, state.labels = state.models_dict[model_path]

    return {"status": "loaded", "model_path": model_path}


@app.post("/infer_model")
async def infer_model_endpoint(file: UploadFile,  top_k: int = 5, request: Request = None):
    state = request.app.state

    image = await uploadfile_to_rgb_image(file)
    top_classes = await asyncio.to_thread(infer_model, state.model, state.labels, image, top_k)
    return {"top_classes": top_classes}
