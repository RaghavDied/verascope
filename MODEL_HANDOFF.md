# VeraScope — Model Handoff (Ishani → Raghav)

## 1. The deliverable file

- **File:** `effnet_b4_deepfake.pt` — https://drive.google.com/file/d/1JPvr_HLEhp0uS6YGLMfN38rk8BT7UeEV/view?usp=sharing
- **Expected file size:** 71.0 MB
- This is a `state_dict` only — weights, not a full pickled model. You need the
  class definition below to load it.
- Evaluation numbers (140k test + FF++ test) are in `final_evaluation.csv`,
  shared alongside this doc.

## 2. The model class — copy this exactly

```python
import timm
import torch.nn as nn

class EfficientNetB4Deepfake(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=1)
    def forward(self, x):
        return self.backbone(x).squeeze(1)
```

Loading it:
```python
import torch
model = EfficientNetB4Deepfake(pretrained=False)
model.load_state_dict(torch.load('effnet_b4_deepfake.pt', map_location=device))
model.eval()
```

The model outputs **one raw logit**, not two class scores. `torch.sigmoid(logit)`
gives you `P(fake)` directly, in `[0, 1]`.

## 3. ⚠️ The part that's easy to get wrong: face-cropping

**The model was trained exclusively on already face-cropped images** — the
140k dataset ships pre-cropped, and the FF++ frames were run through MTCNN
face detection before ever being saved for training. **It has never seen a
full, uncropped photo or video frame during training.**

If the app takes a user's uploaded image/video and feeds the whole frame
into the model directly, predictions will not error out — they'll just be
unreliable, silently. This needs a face-crop step in the pipeline *before*
the model ever sees the image, every time, no exceptions.

The fix is one library, already used on the training side:
```bash
pip install facenet-pytorch
```
```python
from facenet_pytorch import MTCNN
mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, post_process=False, device=device)
face_tensor = mtcnn(pil_image)  # returns None if no face found -- handle that case
```

## 4. Preprocessing spec (must match exactly, or results will be off)

- Resize to **380×380** (EfficientNet-B4's native resolution)
- Normalize with ImageNet mean/std: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- No augmentation at inference time (no flips/rotation/color jitter — those are train-only)

```python
from torchvision import transforms
val_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

## 5. Video handling: frame-level averaging (per the original architecture decision)

Per our agreed design — no separate video model, 12 evenly spaced frames,
classify each with the image model, average the fake-probabilities into one
verdict:

```python
import cv2
import numpy as np

def sample_frame_indices(total_frames, n=12):
    return np.linspace(0, max(total_frames - 1, 0), n, dtype=int)
```
Read each sampled frame with `cv2.VideoCapture`, face-crop it with MTCNN
(step 3), classify it (step 2/4), then average the resulting probabilities
across however many frames actually had a detectable face. If zero frames
had a detectable face, treat that as an error case (see below), not a
silent 0% or 100%.

## 6. Drop-in reference matching your existing `analyze_image()` schema

This mirrors the exact dict shape your dummy version already returns, so
swapping it in shouldn't require touching anything else in `app.py` or the
templates. Treat this as a reference to adapt, not a finished file —
confirm paths and error handling fit how the rest of `app/` is structured
before dropping it in.

```python
import os
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms

from explainability.gradcam import EfficientNetB4Deepfake  # or redefine locally

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'effnet_b4_deepfake.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = EfficientNetB4Deepfake(pretrained=False).to(device)
_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
_model.eval()

_mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, post_process=False, device=device)

_val_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

FRAMES_PER_VIDEO = 12
VIDEO_EXTENSIONS = {'.mp4', '.avi'}


def _crop_face(pil_img):
    """Returns a face-cropped PIL image, or None if no face detected."""
    face_tensor = _mtcnn(pil_img)
    if face_tensor is None:
        return None
    face_np = face_tensor.permute(1, 2, 0).numpy().astype('uint8')
    return Image.fromarray(face_np)


def _classify_face(face_pil):
    """Returns P(fake) in [0, 1] for one already-cropped face image."""
    input_tensor = _val_transform(face_pil.resize((380, 380))).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.sigmoid(_model(input_tensor)).item()


def analyze_image(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in VIDEO_EXTENSIONS:
        fake_probs = []
        cap = cv2.VideoCapture(file_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(total - 1, 0), FRAMES_PER_VIDEO, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face = _crop_face(pil_frame)
            if face is not None:
                fake_probs.append(_classify_face(face))
        cap.release()
        if not fake_probs:
            raise ValueError("No face detected in any sampled frame.")
        fake_pct = round(100 * sum(fake_probs) / len(fake_probs), 1)
    else:
        pil_img = Image.open(file_path).convert('RGB')
        face = _crop_face(pil_img)
        if face is None:
            raise ValueError("No face detected in the uploaded image.")
        fake_pct = round(100 * _classify_face(face), 1)

    real_pct = round(100 - fake_pct, 1)
    prediction = "Fake" if fake_pct > 50 else "Real"
    confidence = fake_pct if prediction == "Fake" else real_pct

    return {
        "prediction": prediction,
        "confidence_pct": confidence,
        "real_pct": real_pct,
        "fake_pct": fake_pct,
        "explanation": (
            f"Model flagged this as {prediction.lower()} with {confidence}% confidence, "
            f"based on {'frame-averaged ' if ext in VIDEO_EXTENSIONS else ''}facial analysis."
        ),
        "recommendation": "Treat with caution" if prediction == "Fake" else "No immediate concerns flagged.",
    }
```

**One decision this leaves for you:** what should the app show the user if
`analyze_image()` raises `ValueError("No face detected...")`? That's a real
case (bad lighting, face turned away, no face in frame) and the dummy
version never had to handle it since it never actually looked at the image.

## 7. Grad-CAM integration

`explainability/gradcam.py` exposes `get_gradcam_heatmap(image_path, model=None, model_path=...)`,
returning `(pil_original, overlay_rgb_array, fake_probability)`. If/when the
UI wants to show the heatmap, pass it the same face-cropped image (not the
raw upload) for consistency with how the model was trained — happy to
adjust the return format (e.g. saved file path instead of a numpy array) if
that's easier on your end, just say which you'd rather have.
