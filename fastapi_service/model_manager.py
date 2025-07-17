from torchvision import models
import os
from utils import save_pickle, load_pickle

MODEL_REGISTRY = {
    "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
}

"""
MODEL_REGISTRY = {
    # AlexNet
    "alexnet": (models.alexnet, models.AlexNet_Weights.IMAGENET1K_V1),

    # VGG
    "vgg11":      (models.vgg11,      models.VGG11_Weights.IMAGENET1K_V1),
    "vgg11_bn":   (models.vgg11_bn,   models.VGG11_BN_Weights.IMAGENET1K_V1),
    "vgg13":      (models.vgg13,      models.VGG13_Weights.IMAGENET1K_V1),
    "vgg13_bn":   (models.vgg13_bn,   models.VGG13_BN_Weights.IMAGENET1K_V1),
    "vgg16":      (models.vgg16,      models.VGG16_Weights.IMAGENET1K_V1),
    "vgg16_bn":   (models.vgg16_bn,   models.VGG16_BN_Weights.IMAGENET1K_V1),
    "vgg19":      (models.vgg19,      models.VGG19_Weights.IMAGENET1K_V1),
    "vgg19_bn":   (models.vgg19_bn,   models.VGG19_BN_Weights.IMAGENET1K_V1),

    # SqueezeNet
    "squeezenet1_0": (models.squeezenet1_0, models.SqueezeNet1_0_Weights.IMAGENET1K_V1),
    "squeezenet1_1": (models.squeezenet1_1, models.SqueezeNet1_1_Weights.IMAGENET1K_V1),

    # DenseNet
    "densenet121": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1),
    "densenet169": (models.densenet169, models.DenseNet169_Weights.IMAGENET1K_V1),
    "densenet201": (models.densenet201, models.DenseNet201_Weights.IMAGENET1K_V1),
    "densenet161": (models.densenet161, models.DenseNet161_Weights.IMAGENET1K_V1),

    # Inception
    "inception_v3": (models.inception_v3, models.Inception_V3_Weights.IMAGENET1K_V1),

    # GoogLeNet
    "googlenet":    (models.googlenet,    models.GoogLeNet_Weights.IMAGENET1K_V1),

    # ShuffleNet V2
    "shufflenet_v2_x0_5": (models.shufflenet_v2_x0_5, models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1),
    "shufflenet_v2_x1_0": (models.shufflenet_v2_x1_0, models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
    "shufflenet_v2_x1_5": (models.shufflenet_v2_x1_5, models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1),
    "shufflenet_v2_x2_0": (models.shufflenet_v2_x2_0, models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1),

    # MobileNet
    "mobilenet_v2":       (models.mobilenet_v2,       models.MobileNet_V2_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V1),
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),

    # MNASNet
    "mnasnet0_5": (models.mnasnet0_5, models.MNASNet0_5_Weights.IMAGENET1K_V1),
    "mnasnet0_75":(models.mnasnet0_75,models.MNASNet0_75_Weights.IMAGENET1K_V1),
    "mnasnet1_0": (models.mnasnet1_0, models.MNASNet1_0_Weights.IMAGENET1K_V1),
    "mnasnet1_3": (models.mnasnet1_3, models.MNASNet1_3_Weights.IMAGENET1K_V1),

    # EfficientNet B0–B7
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.IMAGENET1K_V1),
    "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.IMAGENET1K_V1),
    "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1),
    "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.IMAGENET1K_V1),
    "efficientnet_b5": (models.efficientnet_b5, models.EfficientNet_B5_Weights.IMAGENET1K_V1),
    "efficientnet_b6": (models.efficientnet_b6, models.EfficientNet_B6_Weights.IMAGENET1K_V1),
    "efficientnet_b7": (models.efficientnet_b7, models.EfficientNet_B7_Weights.IMAGENET1K_V1),

    # EfficientNet V2
    "efficientnet_v2_s": (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1),
    "efficientnet_v2_m": (models.efficientnet_v2_m, models.EfficientNet_V2_M_Weights.IMAGENET1K_V1),
    "efficientnet_v2_l": (models.efficientnet_v2_l, models.EfficientNet_V2_L_Weights.IMAGENET1K_V1),

    # RegNet Y
    "regnet_y_400mf": (models.regnet_y_400mf, models.RegNet_Y_400MF_Weights.IMAGENET1K_V1),
    "regnet_y_800mf": (models.regnet_y_800mf, models.RegNet_Y_800MF_Weights.IMAGENET1K_V1),
    "regnet_y_1_6gf": (models.regnet_y_1_6gf, models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V1),
    "regnet_y_3_2gf": (models.regnet_y_3_2gf, models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V1),
    "regnet_y_8gf":   (models.regnet_y_8gf,   models.RegNet_Y_8GF_Weights.IMAGENET1K_V1),

    # RegNet X
    "regnet_x_400mf": (models.regnet_x_400mf, models.RegNet_X_400MF_Weights.IMAGENET1K_V1),
    "regnet_x_800mf": (models.regnet_x_800mf, models.RegNet_X_800MF_Weights.IMAGENET1K_V1),
    "regnet_x_1_6gf": (models.regnet_x_1_6gf, models.RegNet_X_1_6GF_Weights.IMAGENET1K_V1),
    "regnet_x_3_2gf": (models.regnet_x_3_2gf, models.RegNet_X_3_2GF_Weights.IMAGENET1K_V1),

    # Vision Transformers (ViT)
    "vit_b_16":   (models.vit_b_16,   models.ViT_B_16_Weights.IMAGENET1K_V1),
    "vit_b_32":   (models.vit_b_32,   models.ViT_B_32_Weights.IMAGENET1K_V1),
    "vit_l_16":   (models.vit_l_16,   models.ViT_L_16_Weights.IMAGENET1K_V1),
    "vit_l_32":   (models.vit_l_32,   models.ViT_L_32_Weights.IMAGENET1K_V1),
    "vit_h_14":   (models.vit_h_14,   models.ViT_H_14_Weights.IMAGENET1K_V1),

    # Swin Transformer
    "swin_t_patch4_window7_224":  (models.swin_t,  models.Swin_T_Weights.IMAGENET1K_V1),
    "swin_s_patch4_window7_224":  (models.swin_s,  models.Swin_S_Weights.IMAGENET1K_V1),
    "swin_b_patch4_window7_224":  (models.swin_b,  models.Swin_B_Weights.IMAGENET1K_V1),

    # ConvNeXt
    "convnext_tiny":  (models.convnext_tiny,  models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1),
    "convnext_base":  (models.convnext_base,  models.ConvNeXt_Base_Weights.IMAGENET1K_V1),
    "convnext_large": (models.convnext_large, models.ConvNeXt_Large_Weights.IMAGENET1K_V1),
}
"""

def save_model(model_name: str, models_dir: str = "models"):
    """Download specified pretrained model + labels and pickle them to *dirpath*."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name '{model_name}'. Valid: {list(MODEL_REGISTRY)}")

    constructor, weights = MODEL_REGISTRY[model_name]
    model = constructor(weights=weights)  # loads pretrained weights
    labels = weights.meta["categories"]  # list of class strings

    payload = {"model": model, "labels": labels}

    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{model_name}.pkl")

    save_pickle(path, payload)

    print(f"Model + Labels saved to {path}")


def load_model(path: str):
    """Load a pickled (model, labels) bundle created by ``save_model``."""
    loaded = load_pickle(path)
    model = loaded["model"]
    labels = loaded["labels"]
    model.eval()  # set to inference mode
    return model, labels


if __name__ == '__main__':
    # Example usage: download + cache two common ImageNet models.
    save_model('vgg16')
    save_model('resnet18')
