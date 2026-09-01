import random

def analyze_image(file_path):
    """
    DUMMY VERSION.
    Later this will load our real trained model and actually inspect the image.
    For now it just returns a random but realistic-looking result so we can
    build and test the rest of the app (UI, report, etc.) without waiting
    for training to finish.
    """
    fake_pct = round(random.uniform(5, 95), 1)
    real_pct = round(100 - fake_pct, 1)
    prediction = "Fake" if fake_pct > 50 else "Real"
    confidence = fake_pct if prediction == "Fake" else real_pct

    return {
        "prediction": prediction,
        "confidence_pct": confidence,
        "real_pct": real_pct,
        "fake_pct": fake_pct,
        "explanation": "This is a placeholder result from the dummy detector — no real model is running yet.",
        "recommendation": "N/A (dummy mode).",
    }

def analyze_video(file_path):
    """
    DUMMY VERSION for video.
    Later this will sample frames from the video and run analyze_image()
    logic on each one, then aggregate. For now it just returns a random result,
    same shape as analyze_image(), so the app code doesn't need two separate paths.
    """
    fake_pct = round(random.uniform(5, 95), 1)
    real_pct = round(100 - fake_pct, 1)
    prediction = "Fake" if fake_pct > 50 else "Real"
    confidence = fake_pct if prediction == "Fake" else real_pct

    return {
        "prediction": prediction,
        "confidence_pct": confidence,
        "real_pct": real_pct,
        "fake_pct": fake_pct,
        "frames_analyzed": 12,
        "frames_flagged_fake": round(12 * fake_pct / 100),
        "explanation": "This is a placeholder result from the dummy video detector — no real model is running yet.",
        "recommendation": "N/A (dummy mode).",
    }