"""
explainability/gradcam.py

Generates Grad-CAM heatmap overlays for the VeraScope EfficientNet-B4
deepfake classifier. Importable as a module (see get_gradcam_heatmap) or
runnable standalone for a quick check on one image.

IMPORTANT for whoever calls this (e.g. app/inference.py): the input image
should already be a cropped face, matching how the model was trained --
see MODEL_HANDOFF.md. Running this on an uncropped full photo will give
unreliable heatmaps and probabilities.

Usage:
  python gradcam.py --image /path/to/face.jpg \
      --model-path /path/to/effnet_b4_deepfake.pt \
      --output /path/to/overlay.png
"""
import argparse

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
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=1)

    def forward(self, x):
        return self.backbone(x).squeeze(1)


class DeepfakeTarget:
    """The model outputs one raw logit (fake-probability), not per-class
    logits -- Grad-CAM just maximizes that logit directly rather than using
    a class-index target."""

    def __call__(self, model_output):
        return model_output


val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_model(model_path, device):
    model = EfficientNetB4Deepfake(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_gradcam_heatmap(image_path, model=None, model_path=None, device=None):
    """Returns (pil_original, overlay_rgb_array, fake_probability).

    Pass either an already-loaded `model` (cheaper if calling this
    repeatedly), or a `model_path` to load fresh each call.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        if model_path is None:
            raise ValueError("Provide either model or model_path.")
        model = load_model(model_path, device)

    cam = GradCAM(model=model, target_layers=[model.backbone.conv_head])

    pil_img = Image.open(image_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    input_tensor = val_transform(pil_img).unsqueeze(0).to(device)
    rgb_img = np.array(pil_img) / 255.0

    grayscale_cam = cam(input_tensor=input_tensor, targets=[DeepfakeTarget()])[0]
    overlay = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam, use_rgb=True)

    with torch.no_grad():
        prob_fake = torch.sigmoid(model(input_tensor)).item()

    return pil_img, overlay, prob_fake


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
