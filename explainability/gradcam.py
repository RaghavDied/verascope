"""
explainability/gradcam.py

Generates Grad-CAM heatmap overlays for the VeraScope EfficientNet-B4
deepfake classifier. Importable as a module (see get_gradcam_heatmap,
generate_gradcam_overlay) or runnable standalone for a quick check on one image.

IMPORTANT: The input image should be a cropped face, matching how the model
was trained. Running this on an uncropped full photo will give unreliable heatmaps
and probabilities.
"""

import argparse
import os
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

IMG_SIZE = 380
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class EfficientNetB4Deepfake(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=1)

    def forward(self, x):
        return self.backbone(x).squeeze(1)


class DeepfakeTarget:
    """The model outputs one raw logit (fake-probability), not per-class logits.
    Grad-CAM maximizes that logit directly rather than using a class-index target."""

    def __call__(self, model_output):
        return model_output


val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_model(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetB4Deepfake(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_gradcam_heatmap(image_input, model=None, model_path=None, device=None):
    """
    Computes the Grad-CAM heatmap overlay.
    
    Args:
        image_input: str path to image, PIL.Image.Image, or numpy array.
        model: Preloaded EfficientNetB4Deepfake instance (recommended).
        model_path: Path to checkpoint if model is not preloaded.
        device: torch.device.

    Returns:
        (pil_img, overlay_rgb_array, prob_fake)
        - pil_img: PIL Image resized to (IMG_SIZE, IMG_SIZE)
        - overlay_rgb_array: uint8 RGB numpy array with the heatmap overlay
        - prob_fake: float probability in [0, 1]
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        if model_path is None:
            raise ValueError("Provide either model or model_path.")
        model = load_model(model_path, device)

    # Load / convert image_input to PIL Image
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert('RGB')
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input).convert('RGB')
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    pil_img_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
    input_tensor = val_transform(pil_img_resized).unsqueeze(0).to(device)
    rgb_img = np.array(pil_img_resized, dtype=np.float32) / 255.0

    cam = GradCAM(model=model, target_layers=[model.backbone.conv_head])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[DeepfakeTarget()])[0]

    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    with torch.no_grad():
        prob_fake = torch.sigmoid(model(input_tensor)).item()

    return pil_img_resized, overlay, prob_fake


def generate_gradcam_overlay(image_input, model, device=None, output_path=None):
    """
    Helper function to generate and optionally save a Grad-CAM heatmap overlay.
    Returns:
        (overlay_pil, prob_fake)
    """
    _, overlay, prob_fake = get_gradcam_heatmap(image_input, model=model, device=device)
    overlay_pil = Image.fromarray(overlay)
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        overlay_pil.save(output_path)
    return overlay_pil, prob_fake


def main():
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM heatmap overlay for one image.")
    parser.add_argument('--image', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--output', default='gradcam_overlay.png')
    args = parser.parse_args()

    _, overlay, prob_fake = get_gradcam_heatmap(args.image, model_path=args.model_path)
    Image.fromarray(overlay).save(args.output)
    print(f"Fake probability: {prob_fake:.4f}")
    print(f"Overlay saved to: {args.output}")


if __name__ == '__main__':
    main()
