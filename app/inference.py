import random
from PIL import Image, ImageDraw
import os

# def analyze_image(file_path):
#     """
#     DUMMY VERSION.
#     Later this will load our real trained model and actually inspect the image.
#     For now it just returns a random but realistic-looking result so we can
#     build and test the rest of the app (UI, report, etc.) without waiting
#     for training to finish.
#     """
#     fake_pct = round(random.uniform(5, 95), 1)
#     real_pct = round(100 - fake_pct, 1)
#     prediction = "Fake" if fake_pct > 50 else "Real"
#     confidence = fake_pct if prediction == "Fake" else real_pct

#     return {
#         "prediction": prediction,
#         "confidence_pct": confidence,
#         "real_pct": real_pct,
#         "fake_pct": fake_pct,
#         "explanation": "This is a placeholder result from the dummy detector — no real model is running yet.",
#         "recommendation": "N/A (dummy mode).",
#     }

# def analyze_video(file_path):
#     """
#     DUMMY VERSION for video.
#     Later this will sample frames from the video and run analyze_image()
#     logic on each one, then aggregate. For now it just returns a random result,
#     same shape as analyze_image(), so the app code doesn't need two separate paths.
#     """
#     fake_pct = round(random.uniform(5, 95), 1)
#     real_pct = round(100 - fake_pct, 1)
#     prediction = "Fake" if fake_pct > 50 else "Real"
#     confidence = fake_pct if prediction == "Fake" else real_pct

#     return {
#         "prediction": prediction,
#         "confidence_pct": confidence,
#         "real_pct": real_pct,
#         "fake_pct": fake_pct,
#         "frames_analyzed": 12,
#         "frames_flagged_fake": round(12 * fake_pct / 100),
#         "explanation": "This is a placeholder result from the dummy video detector — no real model is running yet.",
#         "recommendation": "N/A (dummy mode).",
#     }

def analyze_image(file_path, heatmap_path=None):
    fake_pct = round(random.uniform(5, 95), 1)
    real_pct = round(100 - fake_pct, 1)
    prediction = "Fake" if fake_pct > 50 else "Real"
    confidence = fake_pct if prediction == "Fake" else real_pct

    heatmap_url = None
    if heatmap_path:
        generate_dummy_heatmap(heatmap_path)
        heatmap_url = "/" + heatmap_path.replace("\\", "/")

    return {
        "prediction": prediction,
        "confidence_pct": confidence,
        "real_pct": real_pct,
        "fake_pct": fake_pct,
        "heatmap_path": heatmap_path,
        "heatmap_url": heatmap_url,
        "explanation": "This is a placeholder result from the dummy detector — no real model is running yet.",
        "recommendation": "N/A (dummy mode).",
    }

def analyze_video(file_path, heatmap_path=None):
    fake_pct = round(random.uniform(5, 95), 1)
    real_pct = round(100 - fake_pct, 1)
    prediction = "Fake" if fake_pct > 50 else "Real"
    confidence = fake_pct if prediction == "Fake" else real_pct

    heatmap_url = None
    if heatmap_path:
        generate_dummy_heatmap(heatmap_path)
        heatmap_url = "/" + heatmap_path.replace("\\", "/")

    return {
        "prediction": prediction,
        "confidence_pct": confidence,
        "real_pct": real_pct,
        "fake_pct": fake_pct,
        "frames_analyzed": 12,
        "frames_flagged_fake": round(12 * fake_pct / 100),
        "heatmap_path": heatmap_path,
        "heatmap_url": heatmap_url,
        "explanation": "This is a placeholder result from the dummy video detector — no real model is running yet.",
        "recommendation": "N/A (dummy mode).",
    }

def generate_dummy_heatmap(save_path):
    """
    DUMMY VERSION. Later this will be a real Grad-CAM heatmap overlay from
    explainability/gradcam.py. For now it just draws a plain placeholder
    image so the app's image-serving and PDF-embedding logic can be built
    and tested end-to-end.
    """
    img = Image.new("RGB", (224, 224), color=(230, 230, 230))
    draw = ImageDraw.Draw(img)
    draw.text((40, 100), "Heatmap\n(placeholder)", fill=(200, 30, 30))
    img.save(save_path)
    return save_path