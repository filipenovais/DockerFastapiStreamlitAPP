from torchvision import transforms
import torch

def infer_model(model, labels, image, top_k=5):
    """Get top-*k* ImageNet-style prediction on a PIL Image using *model*."""
    transform = transforms.Compose([
        transforms.Resize(256),              # keep aspect ratio; shorter side -> 256
        transforms.CenterCrop(224),          # crop to 224x224 (what most pretrained models expect)
        transforms.ToTensor(),               # [0,1], shape (C,H,W)
        transforms.Normalize(                # ImageNet normalization
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    inputs = transform(image).unsqueeze(0)

    # Match input device to model device if possible (handles CPU/GPU transparently)
    try:
        device = next(model.parameters()).device
    except StopIteration:  # model with no parameters
        device = torch.device("cpu")
    inputs = inputs.to(device)

    # Forward pass (no gradients)
    with torch.no_grad():
        outputs = model(inputs)  # logits
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = probs.topk(top_k, dim=1)

    # Get top classifications: (class_name, probability_percent)
    top_classes = [
        (labels[idx] if labels else str(idx), round(float(p * 100), 2))
        for idx, p in zip(top_indices[0].tolist(), top_probs[0].tolist())
    ]

    return top_classes

