import os
import sys
import unittest
import numpy as np
import cv2
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
APP_DIR = os.path.join(PROJECT_ROOT, 'app')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from app import app
from inference import analyze_image, FaceNotFoundError
from video_inference import analyze_video
from report import generate_report_pdf


class TestVeraScopeIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.test_outputs_dir = os.path.join(PROJECT_ROOT, 'test_artifacts')
        os.makedirs(cls.test_outputs_dir, exist_ok=True)

        # Ensure sample test face exists
        cls.sample_image = os.path.join(PROJECT_ROOT, 'overlay_fake.png')
        if not os.path.exists(cls.sample_image):
            parent_img = os.path.join(PROJECT_ROOT, '..', 'overlay_fake.png')
            if os.path.exists(parent_img):
                import shutil
                shutil.copy(parent_img, cls.sample_image)

        # Create a small test video with 24 frames containing the sample face
        cls.sample_video = os.path.join(cls.test_outputs_dir, 'sample_test_vid.mp4')
        face_pil = Image.open(cls.sample_image).convert('RGB').resize((240, 240))
        face_cv = cv2.cvtColor(np.array(face_pil), cv2.COLOR_RGB2BGR)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cls.sample_video, fourcc, 10.0, (320, 240))
        for _ in range(20):
            canvas = np.zeros((240, 320, 3), dtype=np.uint8)
            canvas[:, 40:280] = face_cv
            out.write(canvas)
        out.release()

    def test_01_home_page(self):
        """Test GET / returns 200 with VeraScope brand"""
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'VeraScope', resp.data)

    def test_02_image_inference_and_gradcam(self):
        """Test real EfficientNet-B4 + MTCNN + Grad-CAM inference on test face"""
        self.assertTrue(os.path.exists(self.sample_image), "overlay_fake.png should exist")
        heatmap_path = os.path.join(self.test_outputs_dir, 'test_heatmap.jpg')
        face_path = os.path.join(self.test_outputs_dir, 'test_face.jpg')

        result = analyze_image(self.sample_image, heatmap_path=heatmap_path, face_path=face_path)

        self.assertIn("prediction", result)
        self.assertIn(result["prediction"], ["Real", "Fake"])
        self.assertGreaterEqual(result["confidence_pct"], 50.0)
        self.assertTrue(os.path.exists(heatmap_path))
        self.assertGreater(os.path.getsize(heatmap_path), 1000)
        self.assertTrue(os.path.exists(face_path))
        print(f"\n[PASS] Image Inference: {result['prediction']} ({result['confidence_pct']}% confidence)")

    def test_03_video_inference(self):
        """Test multi-frame video inference pipeline"""
        heatmap_path = os.path.join(self.test_outputs_dir, 'test_vid_heatmap.jpg')
        face_path = os.path.join(self.test_outputs_dir, 'test_vid_face.jpg')

        result = analyze_video(self.sample_video, heatmap_path=heatmap_path, face_path=face_path, num_frames=6)

        self.assertIn("prediction", result)
        self.assertGreater(result["frames_analyzed"], 0)
        self.assertTrue(os.path.exists(heatmap_path))
        print(f"\n[PASS] Video Inference: {result['prediction']} ({result['frames_analyzed']} frames analyzed)")

    def test_04_pdf_report_generation(self):
        """Test generation of forensic PDF verification report"""
        heatmap_path = os.path.join(self.test_outputs_dir, 'test_heatmap.jpg')
        face_path = os.path.join(self.test_outputs_dir, 'test_face.jpg')
        pdf_path = os.path.join(self.test_outputs_dir, 'test_report.pdf')

        dummy_result = {
            "prediction": "Fake",
            "confidence_pct": 98.4,
            "real_pct": 1.6,
            "fake_pct": 98.4,
            "heatmap_path": heatmap_path,
            "face_path": face_path,
            "explanation": "High level of synthesis artifacts detected along facial perimeter.",
            "recommendation": "Do not authenticate without secondary forensic checks.",
        }

        generate_report_pdf(dummy_result, "test_evidence.jpg", "2026-10-16 02:00:00", "image", pdf_path)
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 5000)
        print(f"\n[PASS] PDF Report Generated: {os.path.getsize(pdf_path)} bytes")

    def test_05_flask_upload_analyze_endpoint(self):
        """Test POST /analyze endpoint with multipart file upload"""
        with open(self.sample_image, 'rb') as fp:
            data = {'media': (fp, 'test_upload.png')}
            resp = self.client.post('/analyze', data=data, content_type='multipart/form-data')
            self.assertEqual(resp.status_code, 200)
            self.assertIn(b'Verification Assessment', resp.data)
            self.assertIn(b'Grad-CAM', resp.data)
        print("\n[PASS] Flask /analyze endpoint succeeded")


if __name__ == '__main__':
    unittest.main()
