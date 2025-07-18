import asyncio
from fastapi import FastAPI, UploadFile, Request 

from model_manager import MODEL_PARAMS, download_model, load_model
from model_inference import infer_model
from utils import uploadfile_to_rgb_image  

# Initialize the FastAPI application
app = FastAPI()
# Runtime containers for managing model states
app.state.models_dict = {}  # Cache for loaded models: {model_path: (model, labels)}
app.state.model = None      # Currently active model object
app.state.labels = None     # Labels corresponding to the active model

"""Endpoint to load or switch to a model by path (without .pkl extension)"""
@app.get("/load_model")
def load_model_endpoint(model_path: str, request: Request):
    state = request.app.state

    if model_path not in state.models_dict:
        # Load model from disk and store in cache
        model, labels = load_model(f"{model_path}.pkl")
        state.models_dict[model_path] = (model, labels)
    else:
        # Retrieve the model and labels from cache
        state.model, state.labels = state.models_dict[model_path]

    return {"status": "loaded", "model_path": model_path}

"""Endpoint to perform inference on an uploaded image, returning top-k predictions"""
@app.post("/infer_model")
async def infer_model_endpoint(
    file: UploadFile,
    top_k: int = 5,
    request: Request = None,
):
    state = request.app.state
    # Convert the uploaded file into an RGB image array
    image = await uploadfile_to_rgb_image(file)
    # Run inference in a separate thread to avoid blocking
    top_classes = await asyncio.to_thread(
        infer_model,
        state.model,
        state.labels,
        image,
        top_k,
    )
    return {"top_classes": top_classes}

"""Endpoint to download and save a model file to a specified directory"""
@app.get("/download_model")
def download_model_endpoint(models_dir: str, model_name: str):
    model_path = download_model(models_dir, model_name)
    return {"status": "downloaded", "model_path": model_path}

"""Endpoint to list supported torchvision model parameters"""
@app.get("/torchvision_models")
def torchvision_models_endpoint():
    return MODEL_PARAMS
