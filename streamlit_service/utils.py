import os

def get_model_size_mb(params_count):
    """Convert parameter count to approximate size in MB."""
    return round(params_count * 4 / (1024 * 1024), 1)  # 4 bytes per parameter (float32)


def format_model_info(model_name, params_count):
    """Format model information for display."""
    size_mb = get_model_size_mb(params_count)
    return f"{model_name} ({params_count:,} params, ~{size_mb} MB)"

def get_existing_models(models_dir):
    """Get list of existing model files in the directory."""
    if not os.path.exists(models_dir):
        return []
    
    pkl_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    return pkl_files
