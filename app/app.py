import os
import sys
import datetime
import uuid
from flask import Flask, render_template, request, send_from_directory, redirect, url_for

# Ensure both app directory and project root are in sys.path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))
for p in (APP_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from inference import analyze_image, FaceNotFoundError
from video_inference import analyze_video
from report import generate_report_pdf

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")
OUTPUTS_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

IMAGE_EXTS = {"jpg", "jpeg", "png"}
VIDEO_EXTS = {"mp4", "avi", "mov"}
ALLOWED_EXTS = IMAGE_EXTS | VIDEO_EXTS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "media" not in request.files:
        return render_template("index.html", error="No media file provided. Please choose a file to analyze.")

    file = request.files["media"]

    if not file or file.filename == "" or "." not in file.filename:
        return render_template("index.html", error="Please choose a valid file.")

    ext = file.filename.rsplit(".", 1)[1].lower()

    if ext not in ALLOWED_EXTS:
        return render_template(
            "index.html",
            error=f"Unsupported file format (.{ext}). Allowed types: Images ({', '.join(IMAGE_EXTS)}) or Videos ({', '.join(VIDEO_EXTS)})."
        )

    # Generate a unique job identifier
    job_id = uuid.uuid4().hex[:8]
    safe_filename = f"{job_id}_{os.path.basename(file.filename)}"
    save_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    file.save(save_path)

    upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    heatmap_path = os.path.join(OUTPUTS_FOLDER, f"{job_id}_heatmap.jpg")
    face_path = os.path.join(OUTPUTS_FOLDER, f"{job_id}_face.jpg")
    report_path = os.path.join(REPORTS_FOLDER, f"{job_id}.pdf")

    try:
        if ext in IMAGE_EXTS:
            result = analyze_image(
                save_path,
                output_dir=OUTPUTS_FOLDER,
                job_id=job_id,
                heatmap_path=heatmap_path,
                face_path=face_path,
            )
            media_type = "image"
        else:
            result = analyze_video(save_path, heatmap_path=heatmap_path, face_path=face_path)
            media_type = "video"

        # Generate Evidence Verification PDF Report
        generate_report_pdf(
            result=result,
            filename=file.filename,
            upload_time=upload_time,
            media_type=media_type,
            out_path=report_path,
        )

        return render_template(
            "result.html",
            result=result,
            media_type=media_type,
            upload_time=upload_time,
            filename=file.filename,
            report_id=job_id,
        )

    except FaceNotFoundError as fnfe:
        return render_template(
            "index.html",
            error=str(fnfe),
            warning_level="warning"
        )
    except Exception as e:
        return render_template(
            "index.html",
            error=f"An error occurred during forensic analysis: {str(e)}"
        )


@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUTS_FOLDER, filename)


@app.route("/download_report/<report_id>")
def download_report(report_id):
    report_file = f"{report_id}.pdf"
    file_path = os.path.join(REPORTS_FOLDER, report_file)
    if not os.path.exists(file_path):
        return "Report not found or expired.", 404
    return send_from_directory(
        REPORTS_FOLDER,
        report_file,
        as_attachment=True,
        download_name=f"VeraScope_Verification_{report_id}.pdf"
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)