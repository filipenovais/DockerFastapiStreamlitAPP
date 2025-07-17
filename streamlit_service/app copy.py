import streamlit as st
import os
import requests
from PIL import Image

st.set_page_config(layout="centered")
image_dir = "/images"
model_dir = "/models"

tab1, tab2 = st.tabs(["🖼️ Classify", "🛠️ Manage Models"])

def infer(image_path):
    with open(image_path, 'rb') as f:
        files = {"file": f}
        res = requests.post(f"http://fastapi:8000/infer", files=files)
        return res.json()

def load_model(selected_model):
    res = requests.get(f"http://fastapi:8000/load", params={"model_path": f"/models/{selected_model}"})
    return res.json()


images = sorted(os.listdir(image_dir))
models = os.listdir(model_dir)

with tab1:
    col_screen, col_controls = st.columns([2, 1])

    col_screen.header("Image Classification")

    selected_model = col_controls.selectbox("Select Model", models)
    # load automatically without button
    if col_controls.button("Load Selected Model"):
        result = load_model(selected_model)
    # add something to chcek if model was load (red for error green for good)
        
    selected_image = col_controls.radio("Select Image", images)

    image_path = os.path.join(image_dir, selected_image)
    image = Image.open(image_path)
    col_screen.image(image, width=250)

    if col_screen.button("Classify"):
        result = infer(image_path)  # Fixed port for simplicity
        col_screen.write("Top 5 Classes:", result["top5"])

with tab2:
    st.header("Manage Models")
    st.write("Available:", models)
