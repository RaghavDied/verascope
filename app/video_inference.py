import random
import cv2


def sample_frames(video_path, num_frames=12):
    """
    Extract up to num_frames evenly spaced frames from the video.
    This is the real logic (not dummy) — it will keep working unchanged
    once Phase 2's trained model is dropped into inference.py, since
    analyze_video() will just swap the random score for a real model call
    per frame.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return frames

    indices = sorted(set(int(i * total / num_frames) for i in range(num_frames)))
    idx_set = set(indices)

    current = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current in idx_set:
            frames.append(frame)
        current += 1
        if current > total:
            break

    cap.release()
    return frames


def analyze_video(file_path, num_frames=12):
    """
    DUMMY VERSION — frame-level video analysis.

    Real pipeline (Phase 3, once Ishani's checkpoint exists):
      1. sample_frames() pulls evenly spaced frames (already real, above)
      2. each frame gets face-cropped (same preprocessing as Module 3)
      3. each cropped frame runs through the trained EfficientNet-B4
         model from inference.py (reused, no separate video model)
      4. per-frame fake-probabilities are averaged into one verdict

    For now, step 3 is replaced with a random score per frame so the
    UI/report pipeline can be built and demoed without waiting on training.
    """
    frames = sample_frames(file_path, num_frames=num_frames)
    # fall back to num_frames so dummy mode still works even on a
    # corrupt/unreadable file, rather than crashing the demo
    n = len(frames) if frames else num_frames

    frame_scores = [round(random.uniform(5, 95), 1) for _ in range(n)]
    avg_fake_pct = round(sum(frame_scores) / len(frame_scores), 1)
    real_pct = round(100 - avg_fake_pct, 1)
    prediction = "Fake" if avg_fake_pct > 50 else "Real"
    confidence = avg_fake_pct if prediction == "Fake" else real_pct

    suspicious_frame_count = sum(1 for s in frame_scores if s > 50)

    return {
        "prediction": prediction,
        "confidence_pct": confidence,
        "real_pct": real_pct,
        "fake_pct": avg_fake_pct,
        "frames_analyzed": n,
        "suspicious_frame_count": suspicious_frame_count,
        "frame_scores": frame_scores,
        "explanation": (
            f"This is a placeholder result from the dummy video detector — "
            f"no real model is running yet. {suspicious_frame_count}/{n} sampled "
            f"frames were flagged above the 50% fake threshold."
        ),
        "recommendation": "N/A (dummy mode).",
    }

