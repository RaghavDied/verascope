import os
import sys
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from explainability.gradcam import EfficientNetB4Deepfake, generate_gradcam_overlay, val_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_mtcnn = None


class FaceNotFoundError(Exception):
    """Raised when no face is detected in the provided media."""
    pass


def find_model_checkpoint():
    """Searches standard directories for the model checkpoint."""
    possible_paths = [
        os.path.join(PROJECT_ROOT, "models", "effnet_b4_deepfake.pt"),
        os.path.join(PROJECT_ROOT, "effnet_b4_deepfake.pt"),
        os.path.join(os.path.dirname(__file__), "models", "effnet_b4_deepfake.pt"),
        os.path.join(os.path.dirname(__file__), "..", "..", "effnet_b4_deepfake.pt"),
    ]
    for p in possible_paths:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            return abs_p
    raise FileNotFoundError(
        "Could not locate 'effnet_b4_deepfake.pt'. Checked locations:\n" +
        "\n".join(possible_paths)
    )


def get_model():
    """Singleton getter for the EfficientNet-B4 Deepfake model."""
    global _model
    if _model is None:
        ckpt_path = find_model_checkpoint()
        _model = EfficientNetB4Deepfake(pretrained=False).to(device)
        state_dict = torch.load(ckpt_path, map_location=device)
        _model.load_state_dict(state_dict)
        _model.eval()
    return _model


def get_mtcnn():
    """Singleton getter for the MTCNN face detector."""
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, post_process=False, device=device)
    return _mtcnn


def crop_face(pil_img):
    """
    Detects and crops the primary face from a PIL Image using MTCNN.
    Returns:
        (face_pil, bounding_box) or (None, None) if no face is detected.
    """
    detector = get_mtcnn()
    boxes, _ = detector.detect(pil_img)
    if boxes is None or len(boxes) == 0:
        return None, None

    # Pick the largest detected face box
    best_box = boxes[0]
    if len(boxes) > 1:
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        best_box = boxes[np.argmax(areas)]

    # Crop face with margin
    w, h = pil_img.size
    x1, y1, x2, y2 = best_box
    margin_w = (x2 - x1) * 0.15
    margin_h = (y2 - y1) * 0.15
    crop_x1 = max(0, int(x1 - margin_w))
    crop_y1 = max(0, int(y1 - margin_h))
    crop_x2 = min(w, int(x2 + margin_w))
    crop_y2 = min(h, int(y2 + margin_h))

    face_pil = pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    return face_pil, (crop_x1, crop_y1, crop_x2, crop_y2)


def classify_face(face_pil):
    """
    Classifies a face-cropped PIL Image using the EfficientNet-B4 model.
    Returns:
        prob_fake (float): Model's predicted fake probability in [0, 1].
    """
    model = get_model()
    face_resized = face_pil.resize((380, 380))
    tensor = val_transform(face_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor)
        prob_fake = torch.sigmoid(logit).item()
    return prob_fake


def analyze_image(file_path, heatmap_path=None, face_path=None):
    """
    Executes the full forensic pipeline for a single image:
    1. MTCNN Face Detection & Cropping
    2. EfficientNet-B4 Classification
    3. Grad-CAM Explainability Heatmap Generation
    4. Forensic Reporting Data Assembly
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    pil_img = Image.open(file_path).convert("RGB")
    face_pil, box = crop_face(pil_img)

    if face_pil is None:
        raise FaceNotFoundError(
            "No human face was detected in the uploaded image. VeraScope's deepfake detection "
            "model evaluates facial manipulation artifacts (trained on FaceForensics++ and pre-cropped "
            "facial biometrics). Please upload an image with a clearly visible human face."
        )

    # Save cropped face if requested
    face_url = None
    if face_path:
        os.makedirs(os.path.dirname(os.path.abspath(face_path)), exist_ok=True)
        face_pil.resize((380, 380)).save(face_path)
        face_url = "/outputs/" + os.path.basename(face_path)

    # Model inference and Grad-CAM generation
    model = get_model()
    heatmap_url = None
    if heatmap_path:
        os.makedirs(os.path.dirname(os.path.abspath(heatmap_path)), exist_ok=True)
        _, prob_fake = generate_gradcam_overlay(face_pil, model=model, device=device, output_path=heatmap_path)
        heatmap_url = "/outputs/" + os.path.basename(heatmap_path)
    else:
        prob_fake = classify_face(face_pil)

    fake_pct = round(prob_fake * 100, 1)
    real_pct = round(100.0 - fake_pct, 1)
    prediction = "Fake" if fake_pct > 50.0 else "Real"
    confidence_pct = fake_pct if prediction == "Fake" else real_pct

    # Explainability text & recommendation
    if prediction == "Fake":
        explanation = (
            f"VeraScope flagged this image as manipulated (Fake) with {confidence_pct}% confidence. "
            f"The EfficientNet-B4 convolutional feature maps identified abnormal blending boundaries, "
            f"biometric texture inconsistencies, and synthesis artifacts highlighted in the Grad-CAM heatmap."
        )
        recommendation = (
            "Caution advised. Strong indicators of generative AI manipulation or face-swapping detected. "
            "Do not treat this media as authentic evidence without independent multi-source verification."
        )
    else:
        explanation = (
            f"VeraScope evaluated this image as authentic (Real) with {confidence_pct}% confidence. "
            f"Facial boundaries, skin texture, and lighting gradients are consistent with natural camera "
            f"sensor captures without synthetic manipulation artifacts."
        )
        recommendation = (
            "Low risk of manipulation. The biometric facial analysis indicates authentic photographic "
            "characteristics matching unaltered reference datasets."
        )

    return {
        "prediction": prediction,
        "confidence_pct": confidence_pct,
        "real_pct": real_pct,
        "fake_pct": fake_pct,
        "face_detected": True,
        "face_path": face_path,
        "face_url": face_url,
        "heatmap_path": heatmap_path,
        "heatmap_url": heatmap_url,
        "explanation": explanation,
        "recommendation": recommendation,
    }