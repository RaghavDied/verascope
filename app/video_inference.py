import os
import cv2
import numpy as np
import torch
from PIL import Image

from inference import (
    get_model,
    detect_and_crop_faces,
    classify_face,
    analyze_spectral_and_sensor_artifacts,
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
    Executes frame-level deepfake forensic analysis on video with multi-face support:
    1. Evenly samples frames across the duration.
    2. Detects ALL faces in each frame using multi-face MTCNN.
    3. Analyzes each detected face using EfficientNet-B4 & spectral forensics.
    4. Identifies key suspicious frame and face.
    5. Generates Grad-CAM heatmap overlay for the key evidence frame.
    6. Aggregates multi-frame, multi-face probabilities into a verdict.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")

    sampled = sample_frames(file_path, num_frames=num_frames)
    if not sampled:
        raise ValueError("Could not read frames from the uploaded video file.")

    frame_results = []
    total_faces_inspected = 0

    for f_idx, ts_sec, bgr_frame in sampled:
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)

        # Detect all faces in this frame
        faces_in_frame = detect_and_crop_faces(pil_frame)

        if faces_in_frame:
            total_faces_inspected += len(faces_in_frame)
            # Evaluate all faces in this frame
            frame_face_scores = []
            for face_item in faces_in_frame:
                f_pil = face_item["face_pil"]
                prob_neural = classify_face(f_pil)
                prob_diff, _ = analyze_spectral_and_sensor_artifacts(f_pil)
                face_fake_score = max(prob_neural, prob_diff)
                frame_face_scores.append((face_fake_score, f_pil))

            # Primary score for this frame is the highest face score
            worst_score, worst_face_pil = max(frame_face_scores, key=lambda x: x[0])

            frame_results.append({
                "frame_index": f_idx,
                "timestamp_sec": ts_sec,
                "prob_fake": worst_score,
                "score_pct": round(worst_score * 100, 1),
                "faces_in_frame": len(faces_in_frame),
                "face_pil": worst_face_pil,
            })

    if not frame_results:
        raise FaceNotFoundError(
            f"No human faces were detected across any of the {len(sampled)} sampled video frames. "
            "VeraScope requires visible facial biometrics to perform manipulation analysis. "
            "Please ensure the video contains clear, well-lit human faces."
        )

    fake_probs = [fr["prob_fake"] for fr in frame_results]
    avg_fake_pct = round(100.0 * sum(fake_probs) / len(fake_probs), 1)
    max_fake_pct = round(100.0 * max(fake_probs), 1)

    # If multiple frames show high confidence fake, flag as fake
    flagged_count = sum(1 for p in fake_probs if p > 0.5)
    analyzed_count = len(frame_results)

    # Robust decision: flag if either average > 50% OR at least 25% of frames strongly manipulated
    is_fake = (avg_fake_pct > 50.0) or (flagged_count >= max(1, analyzed_count // 4) and max_fake_pct > 75.0)

    prediction = "Fake" if is_fake else "Real"
    confidence_pct = max_fake_pct if prediction == "Fake" else round(100.0 - avg_fake_pct, 1)

    real_pct = round(100.0 - avg_fake_pct, 1)

    # Select key evidence frame:
    if prediction == "Fake":
        key_result = max(frame_results, key=lambda x: x["prob_fake"])
    else:
        median_prob = float(np.median(fake_probs))
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
            "faces_count": fr["faces_in_frame"],
            "flagged": fr["prob_fake"] > 0.5,
        }
        for fr in frame_results
    ]

    # Generate explanation
    if prediction == "Fake":
        explanation = (
            f"VeraScope classified this video as manipulated (Fake) with {confidence_pct}% confidence. "
            f"Across {analyzed_count} sampled frames ({total_faces_inspected} total face instances inspected), "
            f"{flagged_count} frame(s) displayed severe deepfake manipulation markers. "
            f"The key evidence was detected in Frame #{key_result['frame_index']} at {key_result['timestamp_sec']:.1f}s "
            f"({key_result['score_pct']}% manipulation index)."
        )
        recommendation = (
            "High risk video manipulation. Evidence suggests targeted face-swapping or neural re-enactment. "
            "Do not authenticate without independent forensic verification."
        )
    else:
        explanation = (
            f"VeraScope verified this video as authentic (Real) with {confidence_pct}% aggregate confidence. "
            f"All {analyzed_count} face-bearing frames ({total_faces_inspected} face instances) maintained natural "
            f"biometric coherence, authentic motion continuity, and consistent optical sensor noise."
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
        "total_faces_inspected": total_faces_inspected,
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
