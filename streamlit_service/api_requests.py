import requests

# URL for the FastAPI backend service
FASTAPI_URL = "http://fastapi:8000"

def infer_model(image_path, top_k):
    """Send an image file for classification; return backend JSON."""
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{FASTAPI_URL}/infer_model", 
                files=files, 
                params={"top_k": top_k}, 
                timeout=30
            )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def load_model(models_dir, selected_model):
    """Request backend to load the chosen model; return backend JSON."""
    try:
        response = requests.get(
            f"{FASTAPI_URL}/load_model", 
            params={"model_path": f"{models_dir}/{selected_model}"}, 
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_torchvision_models():
    """Get available TorchVision models from backend; return backend JSON."""
    try:
        response = requests.get(f"{FASTAPI_URL}/torchvision_models", timeout=30)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def download_model(models_dir, model_name):
    """Download a model from TorchVision; return backend JSON."""
    try:
        response = requests.get(
            f"{FASTAPI_URL}/download_model", 
            params={"models_dir": models_dir, "model_name": model_name}, 
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}