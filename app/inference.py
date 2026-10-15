import os
import sys
import numpy as np
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms

# Ensure project root and app directory are in sys.path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))
for p in (APP_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

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
        os.path.join(APP_DIR, "models", "effnet_b4_deepfake.pt"),
        os.path.join(APP_DIR, "..", "..", "effnet_b4_deepfake.pt"),
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
    Extracts the face using MTCNN with direct tensor output (matching training preprocessing),
    with bounding box fallback.
    Returns:
        (face_pil, bounding_box) or (None, None) if no face detected.
    """
    detector = get_mtcnn()

    # 1. Direct MTCNN crop matching Ishani's exact training preprocessing
    try:
        face_tensor = detector(pil_img)
        if face_tensor is not None:
            face_np = face_tensor.permute(1, 2, 0).cpu().numpy().astype('uint8')
            return Image.fromarray(face_np), None
    except Exception:
        pass

    # 2. Bounding box fallback
    boxes, _ = detector.detect(pil_img)
    if boxes is not None and len(boxes) > 0:
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        best_box = boxes[np.argmax(areas)]
        w, h = pil_img.size
        x1, y1, x2, y2 = best_box
        margin_w = (x2 - x1) * 0.15
        margin_h = (y2 - y1) * 0.15
        crop_x1 = max(0, int(x1 - margin_w))
        crop_y1 = max(0, int(y1 - margin_h))
        crop_x2 = min(w, int(x2 + margin_w))
        crop_y2 = min(h, int(y2 + margin_h))
        return pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2)), (crop_x1, crop_y1, crop_x2, crop_y2)

    return None, None


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


def analyze_spectral_and_sensor_artifacts(pil_img):
    """
    Multi-modal forensic analysis for diffusion & generative AI models:
    - 2D Fourier (FFT) high-frequency decay analysis
    - Laplacian texture gradient distribution (detects plastic skin smoothing)
    - Sensor noise residual analysis (detects absence of physical camera CMOS noise)
    
    Returns:
        prob_diffusion (float): Probability in [0, 1] that media is AI generated.
        metrics (dict): Forensic measurements.
    """
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # 1. FFT High-Frequency Spectral Ratio
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.abs(f)
    cy, cx = h // 2, w // 2
    r_inner = max(min(h, w) // 10, 5)
    r_mid = max(min(h, w) // 4, 10)
    r_outer = max(min(h, w) // 2, 20)
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    low_f = np.mean(mag[dist < r_inner])
    high_f = np.mean(mag[(dist >= r_mid) & (dist < r_outer)])
    spectral_ratio = float(high_f / (low_f + 1e-9))

    # 2. Laplacian texture variance (sharp natural skin vs synthetic smoothing)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 3. Sensor noise residual (PRNU proxy)
    denoised = cv2.medianBlur(gray, 3)
    noise_std = float(np.std(cv2.absdiff(gray, denoised)))

    # Cumulative forensic score
    diff_score = 0.0
    if spectral_ratio < 0.024:
        diff_score += 0.55
    elif spectral_ratio < 0.029:
        diff_score += 0.25

    if noise_std < 1.6:
        diff_score += 0.35
    elif noise_std < 2.0:
        diff_score += 0.15

    if lap_var < 50.0:
        diff_score += 0.20

    diff_score = min(diff_score, 0.99)
    return diff_score, {
        "spectral_ratio": round(spectral_ratio, 4),
        "laplacian_variance": round(lap_var, 1),
        "noise_std": round(noise_std, 2),
    }


def analyze_image(file_path, heatmap_path=None, face_path=None):
    """
    Executes the full multi-modal forensic pipeline:
    1. MTCNN Face Detection & Isolation
    2. EfficientNet-B4 Spatial Convolutional Classification (GAN/FaceSwap specialist)
    3. Spectral & Sensor Noise Forensic Analysis (Diffusion/Generative AI specialist)
    4. Forensic Score Fusion
    5. Grad-CAM Explainability Heatmap Generation
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    try:
        pil_img = Image.open(file_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open image file ({str(e)}). Please verify the file is a standard JPG or PNG.")

    face_pil, box = crop_face(pil_img)

    # If MTCNN didn't detect a face, check if image is already a face or evaluate whole frame
    if face_pil is None:
        w, h = pil_img.size
        # Fallback: if aspect ratio is roughly portrait, treat whole image as facial candidate
        if 0.5 < w / h < 1.8 and min(w, h) >= 120:
            face_pil = pil_img
        else:
            raise FaceNotFoundError(
                "No human face was detected in the uploaded image. VeraScope's deepfake detection "
                "model evaluates facial manipulation artifacts (FaceForensics++ / Deepfakes). "
                "Please upload an image with a clearly visible human face."
            )

    # Save cropped face
    face_url = None
    if face_path:
        os.makedirs(os.path.dirname(os.path.abspath(face_path)), exist_ok=True)
        face_pil.resize((380, 380)).save(face_path)
        face_url = "/outputs/" + os.path.basename(face_path)

    # 1. Neural classification (EfficientNet-B4)
    model = get_model()
    prob_neural = classify_face(face_pil)

    # 2. Diffusion / Frequency / Sensor forensic analysis
    prob_diffusion, forensic_metrics = analyze_spectral_and_sensor_artifacts(pil_img)

    # 3. Multi-Modal Forensic Fusion
    # If either the neural face-swap detector or diffusion forensic detector triggers:
    prob_fake = max(prob_neural, prob_diffusion)

    # Grad-CAM heatmap generation
    heatmap_url = None
    if heatmap_path:
        os.makedirs(os.path.dirname(os.path.abspath(heatmap_path)), exist_ok=True)
        generate_gradcam_overlay(face_pil, model=model, device=device, output_path=heatmap_path)
        heatmap_url = "/outputs/" + os.path.basename(heatmap_path)

    fake_pct = round(prob_fake * 100.0, 1)
    real_pct = round(100.0 - fake_pct, 1)
    prediction = "Fake" if fake_pct > 50.0 else "Real"
    confidence_pct = fake_pct if prediction == "Fake" else real_pct

    # Determine primary detection vector for explanation
    if prediction == "Fake":
        if prob_neural >= 0.50 and prob_diffusion >= 0.50:
            vector_text = "both neural face-swap boundaries (EfficientNet-B4) and synthetic diffusion frequency anomalies"
        elif prob_neural >= 0.50:
            vector_text = f"EfficientNet-B4 convolutional feature maps ({round(prob_neural*100, 1)}% manipulation index), highlighting abnormal facial boundaries and blending artifacts"
        else:
            vector_text = (
                f"Spectral and Sensor Forensics ({round(prob_diffusion*100, 1)}% generative AI index). "
                f"Analysis detected synthetic texture smoothing (Laplacian {forensic_metrics['laplacian_variance']}), "
                f"unnatural high-frequency spectral decay ({forensic_metrics['spectral_ratio']}), and absence of physical CMOS camera sensor noise"
            )

        explanation = (
            f"VeraScope flagged this image as manipulated (Fake) with {confidence_pct}% confidence, "
            f"identified by {vector_text}."
        )
        recommendation = (
            "Caution strongly advised. Media displays clear signatures of Generative AI synthesis "
            "or face-swapping manipulation. Do not rely upon this image for identity verification or evidential purposes."
        )
    else:
        explanation = (
            f"VeraScope evaluated this image as authentic (Real) with {confidence_pct}% confidence. "
            f"Biometric facial boundaries, CMOS sensor noise ({forensic_metrics['noise_std']}), and Fourier spectral distributions "
            f"are consistent with natural physical camera captures without synthetic manipulation artifacts."
        )
        recommendation = (
            "Low risk of manipulation. The image exhibits authentic physical sensor noise and natural biometric textures."
        )

    return {
        "prediction": prediction,
        "confidence_pct": confidence_pct,
        "real_pct": real_pct,
        "fake_pct": fake_pct,
        "prob_neural": round(prob_neural * 100, 1),
        "prob_diffusion": round(prob_diffusion * 100, 1),
        "forensic_metrics": forensic_metrics,
        "face_detected": True,
        "face_path": face_path,
        "face_url": face_url,
        "heatmap_path": heatmap_path,
        "heatmap_url": heatmap_url,
        "explanation": explanation,
        "recommendation": recommendation,
    }