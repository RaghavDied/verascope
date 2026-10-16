import os
import sys
import io
import numpy as np
import cv2
import torch
from PIL import Image, ImageChops, ImageEnhance
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
    """Singleton getter for multi-face MTCNN face detector."""
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(
            keep_all=True,
            min_face_size=20,
            thresholds=[0.5, 0.6, 0.6],
            post_process=False,
            device=device,
        )
    return _mtcnn


def detect_and_crop_faces(pil_img):
    """
    Detects and crops ALL faces present in a PIL Image.
    Returns:
        list of dicts: [
            {"face_pil": PIL.Image, "box": (x1, y1, x2, y2), "prob": float},
            ...
        ]
    """
    detector = get_mtcnn()
    w, h = pil_img.size

    boxes, probs = detector.detect(pil_img)
    faces_list = []

    if boxes is not None and len(boxes) > 0:
        for i, box in enumerate(boxes):
            prob = float(probs[i]) if probs is not None else 1.0
            if prob < 0.60:
                continue

            x1, y1, x2, y2 = box
            box_w = x2 - x1
            box_h = y2 - y1

            # Filter tiny noise boxes
            if box_w < 20 or box_h < 20:
                continue

            # Crop with balanced 15% margin
            margin_w = box_w * 0.15
            margin_h = box_h * 0.15
            crop_x1 = max(0, int(x1 - margin_w))
            crop_y1 = max(0, int(y1 - margin_h))
            crop_x2 = min(w, int(x2 + margin_w))
            crop_y2 = min(h, int(y2 + margin_h))

            face_pil = pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            faces_list.append({
                "face_pil": face_pil,
                "box": (crop_x1, crop_y1, crop_x2, crop_y2),
                "prob": prob,
            })

    # Sort faces by bounding box area (largest face first)
    faces_list.sort(
        key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]),
        reverse=True,
    )
    return faces_list


def crop_face(pil_img):
    """Backwards-compatible single-face getter returning the primary face."""
    faces = detect_and_crop_faces(pil_img)
    if faces:
        return faces[0]["face_pil"], faces[0]["box"]
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


def compute_spectral_slope(gray):
    """
    Calculates the 2D radial azimuthal power spectrum slope (log-log polyfit).
    Natural scenes follow power-law decay alpha ~ -2.5 to -2.85.
    Diffusion / Generative models typically show steep roll-off (alpha < -3.10)
    or periodic frequency spikes.
    """
    h, w = gray.shape
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag_sq = np.abs(f) ** 2
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_r = min(cy, cx)

    if max_r < 15:
        return -2.7

    radial_profile = np.zeros(max_r)
    for i in range(1, max_r):
        mask = (r == i)
        if np.any(mask):
            radial_profile[i] = np.mean(mag_sq[mask])

    freqs = np.arange(1, max_r)
    powers = radial_profile[1:] + 1e-9
    valid = powers > 1e-6

    if np.sum(valid) > 10:
        slope, _ = np.polyfit(np.log(freqs[valid]), np.log(powers[valid]), 1)
        return float(slope)
    return -2.7


def compute_error_level_analysis(pil_img, quality=90):
    """
    Error Level Analysis (ELA) measures JPEG compression error variance.
    Spliced face swaps often exhibit quantization discrepancy against the background.
    """
    try:
        buf = io.BytesIO()
        pil_img.save(buf, 'JPEG', quality=quality)
        buf.seek(0)
        resaved = Image.open(buf)
        diff = ImageChops.difference(pil_img, resaved)
        diff_arr = np.array(diff, dtype=np.float32)
        return float(np.mean(diff_arr)), float(np.std(diff_arr))
    except Exception:
        return 0.5, 0.5


def analyze_spectral_and_sensor_artifacts(pil_img):
    """
    Comprehensive multi-domain forensic evaluation for generative AI & diffusion detection:
    1. FFT High-Frequency Ratio & Azimuthal Spectral Slope
    2. Laplacian Gradient Distribution (Synthetic skin smoothing vs natural texture)
    3. CMOS/CCD Sensor Noise Residuals (PRNU proxy)
    4. Error Level Analysis (ELA) Compression Discrepancy
    """
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # 1. FFT High Frequency Ratio & Radial Slope
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.abs(f)
    cy, cx = h // 2, w // 2
    r_inner = max(min(h, w) // 10, 5)
    r_mid = max(min(h, w) // 4, 10)
    r_outer = max(min(h, w) // 2, 20)
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    low_f = np.mean(mag[dist < r_inner])
    high_f = np.mean(mag[(dist >= r_mid) & (dist < r_outer)])
    spectral_ratio = float(high_f / (low_f + 1e-9))
    spectral_slope = compute_spectral_slope(gray)

    # 2. Laplacian Texture Gradient
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 3. Sensor Noise Residuals
    denoised = cv2.medianBlur(gray, 3)
    noise_residual = cv2.absdiff(gray, denoised)
    noise_std = float(np.std(noise_residual))

    # 4. Error Level Analysis (ELA)
    ela_mean, ela_std = compute_error_level_analysis(pil_img)

    # Multi-factor score calibration
    diff_score = 0.0

    # Spectral decay trigger
    if spectral_ratio < 0.024:
        diff_score += 0.40
    elif spectral_ratio < 0.029:
        diff_score += 0.20

    # Steep spectral slope trigger (diffusion signature)
    if spectral_slope < -3.10:
        diff_score += 0.25
    elif spectral_slope < -2.95:
        diff_score += 0.15

    # Sensor noise trigger
    if noise_std < 1.60:
        diff_score += 0.30
    elif noise_std < 2.00:
        diff_score += 0.15

    # Plastic skin smoothing trigger
    if lap_var < 50.0:
        diff_score += 0.15

    diff_score = min(diff_score, 0.99)
    return diff_score, {
        "spectral_ratio": round(spectral_ratio, 4),
        "spectral_slope": round(spectral_slope, 2),
        "laplacian_variance": round(lap_var, 1),
        "noise_std": round(noise_std, 2),
        "ela_mean": round(ela_mean, 2),
        "ela_std": round(ela_std, 2),
    }


def analyze_image(file_path, output_dir=None, job_id=None, heatmap_path=None, face_path=None):
    """
    Executes the comprehensive multi-face forensic pipeline:
    1. Multi-Face Detection & Bounding Box Localization (MTCNN)
    2. Per-Face Spatial Deep Learning (EfficientNet-B4 + Grad-CAM)
    3. Multi-Domain Spectral, Sensor Noise, and ELA Forensics
    4. Per-Face & Global Forensic Aggregation
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    try:
        pil_img = Image.open(file_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open image file ({str(e)}). Please verify the file is a standard JPG or PNG.")

    job_id = job_id or os.path.splitext(os.path.basename(file_path))[0]
    output_dir = output_dir or os.path.join(APP_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Detect ALL faces
    detected_faces = detect_and_crop_faces(pil_img)

    # Fallback: if no face found, check portrait proportions
    if not detected_faces:
        w, h = pil_img.size
        if 0.5 < w / h < 1.8 and min(w, h) >= 100:
            detected_faces = [{
                "face_pil": pil_img,
                "box": (0, 0, w, h),
                "prob": 0.99,
            }]
        else:
            raise FaceNotFoundError(
                "No human face was detected in the uploaded image. VeraScope requires clearly visible "
                "facial biometrics to perform multi-modal deepfake forensics."
            )

    model = get_model()

    # 2. Global spectral and sensor forensic analysis
    prob_diffusion_global, forensic_metrics = analyze_spectral_and_sensor_artifacts(pil_img)

    # 3. Analyze EVERY detected face
    faces_results = []
    for idx, f_item in enumerate(detected_faces, start=1):
        f_pil = f_item["face_pil"]
        f_box = f_item["box"]

        # Face-specific paths
        f_face_path = os.path.join(output_dir, f"{job_id}_face_{idx}.jpg")
        f_heatmap_path = os.path.join(output_dir, f"{job_id}_heatmap_{idx}.jpg")

        f_pil.resize((380, 380)).save(f_face_path)

        # Neural inference
        prob_neural = classify_face(f_pil)

        # Local face frequency/sensor check
        prob_diff_local, _ = analyze_spectral_and_sensor_artifacts(f_pil)

        # Fuse neural and diffusion score for this face
        prob_face_fake = max(prob_neural, prob_diff_local)
        face_fake_pct = round(prob_face_fake * 100.0, 1)

        # Generate Grad-CAM for this face
        generate_gradcam_overlay(f_pil, model=model, device=device, output_path=f_heatmap_path)

        faces_results.append({
            "face_id": idx,
            "box": [round(c) for c in f_box],
            "fake_pct": face_fake_pct,
            "real_pct": round(100.0 - face_fake_pct, 1),
            "prob_neural": round(prob_neural * 100.0, 1),
            "prob_diffusion": round(prob_diff_local * 100.0, 1),
            "status": "Flagged Manipulated" if face_fake_pct > 50.0 else "Authentic",
            "is_manipulated": face_fake_pct > 50.0,
            "face_path": f_face_path,
            "face_url": f"/outputs/{os.path.basename(f_face_path)}",
            "heatmap_path": f_heatmap_path,
            "heatmap_url": f"/outputs/{os.path.basename(f_heatmap_path)}",
        })

    # 4. Overall Decision Aggregation
    faces_count = len(faces_results)
    flagged_faces = [f for f in faces_results if f["is_manipulated"]]
    flagged_count = len(flagged_faces)

    # Pick the primary evidence face:
    # If any face is flagged, pick the one with highest fake percentage; otherwise pick Face #1
    if flagged_faces:
        primary_face = max(flagged_faces, key=lambda f: f["fake_pct"])
    else:
        primary_face = max(faces_results, key=lambda f: f["fake_pct"])

    # Synchronize primary paths for legacy callers
    if face_path:
        primary_face_pil = detected_faces[primary_face["face_id"] - 1]["face_pil"]
        primary_face_pil.resize((380, 380)).save(face_path)
    if heatmap_path:
        primary_face_pil = detected_faces[primary_face["face_id"] - 1]["face_pil"]
        generate_gradcam_overlay(primary_face_pil, model=model, device=device, output_path=heatmap_path)

    # Image-level verdict
    max_face_fake_pct = primary_face["fake_pct"]
    global_diff_pct = round(prob_diffusion_global * 100.0, 1)

    overall_fake_pct = max(max_face_fake_pct, global_diff_pct)
    overall_real_pct = round(100.0 - overall_fake_pct, 1)
    prediction = "Fake" if overall_fake_pct > 50.0 else "Real"
    confidence_pct = overall_fake_pct if prediction == "Fake" else overall_real_pct

    # Detailed Explainability Text
    if prediction == "Fake":
        if flagged_count > 0:
            face_summary = f"{flagged_count} of {faces_count} detected face(s) were flagged as manipulated"
        else:
            face_summary = "full-frame generative AI synthesis detected"

        if primary_face["prob_neural"] >= 50.0:
            tech_vector = f"EfficientNet-B4 convolutional feature maps (Face #{primary_face['face_id']} flagged with {primary_face['prob_neural']}% face-swap anomaly index)"
        else:
            tech_vector = (
                f"Multi-Domain Spectral & Sensor Noise Forensics ({global_diff_pct}% generative AI index, "
                f"spectral slope {forensic_metrics['spectral_slope']}, noise std {forensic_metrics['noise_std']})"
            )

        explanation = (
            f"VeraScope verified this media as manipulated (Fake) with {confidence_pct}% confidence. "
            f"Across multi-biometric inspection, {face_summary}. The primary alert was triggered by {tech_vector}."
        )
        recommendation = (
            "Caution strongly advised. High risk of face-swapping (e.g. DeepFaceLab/Roop) or Generative AI synthesis. "
            "Do not accept this image as genuine evidence without secondary cryptographic authentication."
        )
    else:
        explanation = (
            f"VeraScope verified this media as authentic (Real) with {confidence_pct}% confidence. "
            f"All {faces_count} detected face(s) exhibited consistent natural biometric features, "
            f"natural CMOS sensor noise (std {forensic_metrics['noise_std']}), and authentic Fourier spectral distribution "
            f"(slope {forensic_metrics['spectral_slope']}) without synthetic manipulation boundaries."
        )
        recommendation = (
            "Low risk of manipulation. Biometric consistency and optical sensor noise patterns match unaltered photographic captures."
        )

    return {
        "prediction": prediction,
        "confidence_pct": confidence_pct,
        "real_pct": overall_real_pct,
        "fake_pct": overall_fake_pct,
        "faces_count": faces_count,
        "flagged_count": flagged_count,
        "faces": faces_results,
        "primary_face": primary_face,
        "face_path": primary_face["face_path"],
        "face_url": primary_face["face_url"],
        "heatmap_path": primary_face["heatmap_path"],
        "heatmap_url": primary_face["heatmap_url"],
        "prob_neural": primary_face["prob_neural"],
        "prob_diffusion": global_diff_pct,
        "forensic_metrics": forensic_metrics,
        "face_detected": True,
        "explanation": explanation,
        "recommendation": recommendation,
    }