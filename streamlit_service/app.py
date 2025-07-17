# -------------------------------------------------------------
# Streamlit Frontend: Minimal Image Classification Dashboard
# -------------------------------------------------------------
# Small, targeted comments added for clarity. No functional changes.
# The app lets you:
#   • Pick a model from /models (filenames, stem only)
#   • Pick an image from /images
#   • Send the image to a FastAPI backend for inference
#   • (Tab 2) View which models are available
# NOTE: Error handling is intentionally light so backend issues surface in the UI.
# -------------------------------------------------------------

import os
import requests
import streamlit as st
from PIL import Image
from pathlib import Path
import pandas as pd

# We assume container/working-directory paths /images and /models exist.
# "images" are files in images_dir; "models" are file *stems* in models_dir.
images_dir = "/images"
models_dir = "/models"
images = sorted(os.listdir(images_dir)) if os.path.isdir(images_dir) else []
models = [p.stem for p in Path(models_dir).iterdir() if p.is_file()]
FASTAPIURL = "http://fastapi:8000"

# Backend API helper functions
def infer_model(image_path, top_k):
    """Send an image file for classification; return backend JSON."""
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            r = requests.post(f"{FASTAPIURL}/infer_model", files=files, params={"top_k": top_k}, timeout=30)
        return r.json()
    except Exception as e:  # network error, non-200, etc.
        return {"error": str(e)}


def load_model(selected_model):
    """Request backend to load the chosen model; return backend JSON."""
    try:
        r = requests.get(f"{FASTAPIURL}/load_model", params={"model_path": f"{models_dir}/{selected_model}"}, timeout=30)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.set_page_config(layout="centered", page_title="Image Classifier")

# Tab 1 = end-user classify flow; Tab 2 = quick model listing.
tab1, tab2 = st.tabs(["🏞️ Image Classifier", "🛠️ Manage Models"])

# Track model & image selections across interactions so we don't reload the model unnecessarily when the user re-renders the app.
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
    st.session_state.selected_image = None

with tab1:
    col_screen, col_controls = st.columns([2, 1])
    col_screen.header("Image Classification")
    
    # Model selection (dropdown). We only call backend load when selection changes.
    selected_model = col_controls.selectbox("Select Model", models, index=None)
    if selected_model and selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        resp = load_model(selected_model)
        if "error" in resp:
            col_controls.error(f"Load failed: {resp['error']}")
    
    # Display current selections (model + image) side-by-side.
    col_model_text, col_image_text = col_screen.columns([1, 1])
    col_model_text.write(f"Model: **{st.session_state.selected_model}**")
    col_image_text.write(f"Image: **{st.session_state.selected_image}**")

    # Image selection radio; uses session_state key to persist.
    selected_image = col_controls.selectbox("Select Image", images, key="selected_image")

    # Top-k parameter (not currently wired into backend response; backend returns top5).
    top_k = col_controls.number_input("Number of Top Classes", min_value=1, max_value=100, value=5, step=1, format="%d")

    if selected_image:
        try:
            image_path = os.path.join(images_dir, selected_image)
            image = Image.open(image_path)
            col_screen.image(image)
        except Exception as e:
            col_screen.error(f"Could not open image: {e}")

        # Button to trigger inference call. Disabled unless a model is selected.
        if col_controls.button("👾 Run Model 👾", disabled=not bool(selected_model)):
            result = infer_model(image_path, top_k)  # Fixed port for simplicity
            if "error" in result:
                col_controls.error(result["error"])
            else:
                top_classes = result.get("top_classes", [])
                if top_classes:
                    df = pd.DataFrame(top_classes, columns=["Class", "Score (%)"])
                    st.write("Results:", df)


    
with tab2:
    st.header("Manage Models")
    # Quick diagnostic list of available model file stems.
    st.write("Available:", models)
