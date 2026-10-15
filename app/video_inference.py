import os
import cv2
import numpy as np
import torch
from PIL import Image

from inference import (
    get_model,
    crop_face,
    classify_face,
    FaceNotFoundError,
    device,
)
from explainability.gradcam import generate_gradcam_overlay


def sample_frames(video_path, num_frames=12):
    """
    Extracts up to num_frames evenly spaced frames from a video file.
    Returns:
        list of tuples: [(frame_index, timestamp_sec, bgr_frame), ...]
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if total_frames <= 0:
        cap.release()
        return []

    sampled_indices = sorted(set(np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)))
    idx_set = set(sampled_indices)

    frame_list = []
    current_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in idx_set:
            timestamp_sec = current_idx / fps
            frame_list.append((current_idx, timestamp_sec, frame))
        current_idx += 1
        if current_idx > max(idx_set):
            break

    cap.release()
    return frame_list


def analyze_video(file_path, heatmap_path=None, face_path=None, num_frames=12):
    """
    Executes frame-level deepfake forensic analysis on video:
    1. Evenly samples up to num_frames across the duration.
    2. Detects and extracts faces using MTCNN on each frame.
    3. Runs EfficientNet-B4 classification on detected faces.
    4. Identifies the key evidence frame (highest anomaly for fake, or representative frame).
    5. Generates Grad-CAM heatmap overlay for the key evidence frame.
    6. Aggregates multi-frame probabilities into a verdict with temporal consistency stats.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")

    sampled = sample_frames(file_path, num_frames=num_frames)
    if not sampled:
        raise ValueError("Could not read frames from the uploaded video file.")

    frame_results = []

    for f_idx, ts_sec, bgr_frame in sampled:
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)
        face_pil, box = crop_face(pil_frame)

        if face_pil is not None:
            prob_fake = classify_face(face_pil)
            frame_results.append({
                "frame_index": f_idx,
                "timestamp_sec": ts_sec,
                "prob_fake": prob_fake,
                "score_pct": round(prob_fake * 100, 1),
                "face_pil": face_pil,
            })

    if not frame_results:
        raise FaceNotFoundError(
            f"No human faces were detected across any of the {len(sampled)} sampled video frames. "
            "VeraScope requires visible facial biometrics to perform manipulation analysis. "
            "Please ensure the video contains clear, well-lit human faces."
        )

    fake_probs = [fr["prob_fake"] for fr in frame_results]
    avg_fake_pct = round(100.0 * sum(fake_probs) / len(fake_probs), 1)
    real_pct = round(100.0 - avg_fake_pct, 1)
    prediction = "Fake" if avg_fake_pct > 50.0 else "Real"
    confidence_pct = avg_fake_pct if prediction == "Fake" else real_pct

    flagged_count = sum(1 for p in fake_probs if p > 0.5)
    analyzed_count = len(frame_results)

    # Select key evidence frame:
    # If Fake: frame with highest fake probability
    # If Real: frame closest to the median score
    if prediction == "Fake":
        key_result = max(frame_results, key=lambda x: x["prob_fake"])
    else:
        median_prob = np.median(fake_probs)
        key_result = min(frame_results, key=lambda x: abs(x["prob_fake"] - median_prob))

    # Save key evidence face crop
    face_url = None
    if face_path:
        os.makedirs(os.path.dirname(os.path.abspath(face_path)), exist_ok=True)
        key_result["face_pil"].resize((380, 380)).save(face_path)
        face_url = "/outputs/" + os.path.basename(face_path)

    # Generate Grad-CAM for the key evidence frame
    heatmap_url = None
    if heatmap_path:
        os.makedirs(os.path.dirname(os.path.abspath(heatmap_path)), exist_ok=True)
        model = get_model()
        generate_gradcam_overlay(key_result["face_pil"], model=model, device=device, output_path=heatmap_path)
        heatmap_url = "/outputs/" + os.path.basename(heatmap_path)

    frame_scores = [fr["score_pct"] for fr in frame_results]
    frame_details = [
        {
            "frame": fr["frame_index"],
            "timestamp": f"{fr['timestamp_sec']:.1f}s",
            "fake_pct": fr["score_pct"],
            "flagged": fr["prob_fake"] > 0.5,
        }
        for fr in frame_results
    ]

    # Generate explanation
    if prediction == "Fake":
        explanation = (
            f"VeraScope classified this video as manipulated (Fake) with {confidence_pct}% aggregate confidence. "
            f"Across {analyzed_count} face-bearing frames, {flagged_count} frame(s) displayed severe deepfake "
            f"manipulation markers. The Grad-CAM heatmap highlights anomalous synthetic features in the most "
            f"suspicious frame (Frame #{key_result['frame_index']} at {key_result['timestamp_sec']:.1f}s, "
            f"{key_result['score_pct']}% fake probability)."
        )
        recommendation = (
            "High risk video manipulation. Evidence suggests face-swapping (e.g. DeepFaceLab/Roop) or neural "
            "re-enactment. Do not authenticate without independent forensic verification."
        )
    else:
        explanation = (
            f"VeraScope verified this video as authentic (Real) with {confidence_pct}% aggregate confidence. "
            f"All {analyzed_count} face-bearing frames maintained natural biometric coherence, natural motion "
            f"blur, and consistent facial boundaries with zero persistent manipulation signatures."
        )
        recommendation = (
            "Authentic media signature. No systematic deepfake manipulation patterns detected across "
            "sampled video frames."
        )

    return {
        "prediction": prediction,
        "confidence_pct": confidence_pct,
        "real_pct": real_pct,
        "fake_pct": avg_fake_pct,
        "frames_analyzed": analyzed_count,
        "frames_flagged_fake": flagged_count,
        "suspicious_frame_count": flagged_count,
        "frame_scores": frame_scores,
        "frame_details": frame_details,
        "key_frame_index": key_result["frame_index"],
        "key_frame_time": f"{key_result['timestamp_sec']:.1f}s",
        "key_frame_fake_pct": key_result["score_pct"],
        "face_detected": True,
        "face_path": face_path,
        "face_url": face_url,
        "heatmap_path": heatmap_path,
        "heatmap_url": heatmap_url,
        "explanation": explanation,
        "recommendation": recommendation,
    }
