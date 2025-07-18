import os
import streamlit as st
from PIL import Image
from pathlib import Path
import pandas as pd
from utils import format_model_info, get_existing_models
from api_requests import download_model, load_model, get_torchvision_models, infer_model
# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory paths - assume container/working-directory paths exist
IMAGES_DIR = "/images"
MODELS_DIR = "/models"

# Get available files
images = sorted(os.listdir(IMAGES_DIR)) if os.path.isdir(IMAGES_DIR) else []
models = [p.stem for p in Path(MODELS_DIR).iterdir() if p.is_file()]

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "selected_image" not in st.session_state:
        st.session_state.selected_image = None
    if "df_combined" not in st.session_state:
        st.session_state.df_combined = pd.DataFrame()
    if "torchvision_models" not in st.session_state:
        st.session_state.torchvision_models = get_torchvision_models()
    if "existing_models" not in st.session_state:
        st.session_state.existing_models = get_existing_models(MODELS_DIR)

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_image_classifier_tab():
    """Render the main image classification interface."""
    
    col_screen, col_controls = st.columns([2, 1])
    
    # Model selection
    selected_model = col_controls.selectbox("Select Model", models)
    if selected_model and selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        resp = load_model(MODELS_DIR, selected_model)
        if "error" in resp:
            col_controls.error(f"Load failed: {resp['error']}")
    
    # Display current selections
    col_model_text, col_image_text = col_screen.columns([1, 1])
    col_model_text.write(f"Model: **{st.session_state.selected_model}**")
    col_image_text.write(f"Image: **{st.session_state.selected_image}**")

    # Image selection
    selected_image = col_controls.selectbox("Select Image", images)
    if selected_image and selected_image != st.session_state.selected_image:
        st.session_state.selected_image = selected_image
        st.session_state.df_combined = pd.DataFrame()

    # Top-k parameter
    top_k = col_controls.number_input(
        "Number of Top Classes", 
        min_value=1, 
        max_value=100, 
        value=5, 
        step=1, 
        format="%d"
    )

    # Display selected image
    if st.session_state.selected_image:
        try:
            image_path = os.path.join(IMAGES_DIR, st.session_state.selected_image)
            image = Image.open(image_path)
            col_screen.image(image)
        except Exception as e:
            col_screen.error(f"Could not open image: {e}")

    # Control buttons
    col_refresh, col_run = col_controls.columns([1, 1])
    
    if col_refresh.button("🔄 Refresh", key="refresh_local"):
        st.rerun()

    # Run inference button
    if col_run.button("Run 👾", disabled=not bool(selected_model)):
        handle_inference_request(col_controls, selected_model, top_k)

    # Display results
    if not st.session_state.df_combined.empty:
        df = st.session_state.df_combined.reset_index(drop=True)
        st.write("Results:", df)


def handle_inference_request(col_controls, selected_model, top_k):
    """Handle the inference request and update results."""
    if not st.session_state.selected_image:
        col_controls.error("No image selected.")
        return
    
    image_path = os.path.join(IMAGES_DIR, st.session_state.selected_image)
    result = infer_model(image_path, top_k)
    
    if "error" in result:
        col_controls.error(result["error"])
        return
    
    top_classes = result.get("top_classes", [])
    if not top_classes:
        return
    
    # Remove existing results for this model
    if (selected_model, 'Class') in st.session_state.df_combined.columns:
        st.session_state.df_combined = st.session_state.df_combined.drop(
            selected_model, axis=1, level=0, errors="ignore"
        )
    
    # Add new results
    df = pd.DataFrame(top_classes, columns=["Class", "Score (%)"])
    df.columns = pd.MultiIndex.from_product([[selected_model], df.columns])
    st.session_state.df_combined = pd.concat(
        [st.session_state.df_combined, df], axis=1
    ).dropna(how='all')


def render_settings_tab():
    """Render the settings/management interface."""
    render_images_section()
    render_download_models_section()
    render_existing_models_section()


def render_images_section():
    """Render the images management section."""
    st.header("Images")
    
    uploaded_img = st.file_uploader("Upload Image", type=None, key="upload_image")
    if st.button("➕ Add Image", disabled=uploaded_img is None):
        try:
            dest_path = os.path.join(IMAGES_DIR, uploaded_img.name)
            with open(dest_path, "wb") as out_f:
                out_f.write(uploaded_img.getbuffer())
            st.success(f"Image saved: {uploaded_img.name}")
        except Exception as e:
            st.error(f"Save failed: {e}")


def render_download_models_section():
    """Render the model download section."""
    st.header("Download Models")
    
    # Get all available models
    all_models = list(st.session_state.torchvision_models.keys())
    formatted_options = [
        format_model_info(model, st.session_state.torchvision_models[model]) 
        for model in all_models
    ]
    
    selected_formatted = st.selectbox(
        "Select TorchVision Model", 
        formatted_options,
        help="Choose a model to download"
    )
    
    # Extract the actual model name from the formatted string
    selected_model_download = all_models[formatted_options.index(selected_formatted)]
    
    # Download button
    if st.button("📥 Download", type="primary"):
        if selected_model_download:
            try:
                with st.spinner(f"Downloading {selected_model_download}..."):
                    result = download_model(MODELS_DIR, selected_model_download)
                st.success(f"✅ Model saved to {result['model_path']}")
                st.balloons()
                # Refresh existing models list
                st.session_state.existing_models = get_existing_models(MODELS_DIR)
            except Exception as e:
                st.error(f"❌ Error downloading model: {str(e)}")


def render_existing_models_section():
    """Render the existing models management section."""
    st.header("Available Models")
    
    if not st.session_state.existing_models:
        st.info(f"No models found in {MODELS_DIR}")
        st.markdown("💡 **Tip**: Download some models first!")
        return
    
    # Display each model
    for i, model_file in enumerate(st.session_state.existing_models):
        render_model_item(model_file, i)
    
    # Stats button
    render_stats_button()


def render_model_item(model_file, index):
    """Render a single model item with details and delete button."""
    model_name = model_file.replace('.pkl', '')
    
    # Get file size
    file_path = os.path.join(MODELS_DIR, model_file)
    file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 1)
    
    # Create columns for model info
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.write(f"**{model_name}**")
    with col2:
        st.write(f"{file_size_mb} MB")
    with col3:
        if st.button("🗑️ Delete", key=f"delete_{index}"):
            try:
                os.remove(file_path)
                st.success(f"Deleted {model_file}")
                st.session_state.existing_models = get_existing_models(MODELS_DIR)
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting {model_file}: {str(e)}")
    
    st.divider()


def render_stats_button():
    """Render directory stats button for model management."""
    if st.button("📊 Show Directory Stats"):
        total_size = sum(
            os.path.getsize(os.path.join(MODELS_DIR, f)) 
            for f in st.session_state.existing_models
        )
        total_size_mb = round(total_size / (1024 * 1024), 1)
        
        st.info(f"""
        **Directory Statistics**
        - Total Models: {len(st.session_state.existing_models)}
        - Total Size: {total_size_mb} MB
        """)


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """Main application entry point."""
    st.set_page_config(layout="centered", page_title="Image Classifier")
    
    # Initialize session state
    initialize_session_state()
    
    # Create tabs
    tab1, tab2 = st.tabs(["🏞️ Image Classifier", "🛠️ Settings"])
    
    with tab1:
        render_image_classifier_tab()
    
    with tab2:
        render_settings_tab()


if __name__ == "__main__":
    main()