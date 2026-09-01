# from flask import Flask

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return "VeraScope is alive."

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)





# from flask import Flask, render_template, request
# import os

# app = Flask(__name__)

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/analyze", methods=["POST"])
# def analyze():
#     file = request.files["media"]
#     save_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(save_path)
#     return f"Got your file: {file.filename}. Saved successfully."

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)



# from flask import Flask, render_template, request
# import os
# from inference import analyze_image

# app = Flask(__name__)

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/analyze", methods=["POST"])
# def analyze():
#     file = request.files["media"]
#     save_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(save_path)

#     result = analyze_image(save_path)

#     return f"""
#     <h2>{result['prediction']} — {result['confidence_pct']}% confidence</h2>
#     <p>Real: {result['real_pct']}% | Fake: {result['fake_pct']}%</p>
#     <p>{result['explanation']}</p>
#     <p><i>{result['recommendation']}</i></p>
#     """

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)

# from flask import Flask, render_template, request
# import os
# from inference import analyze_image

# app = Flask(__name__)

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/analyze", methods=["POST"])
# def analyze():
#     file = request.files["media"]
#     save_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(save_path)

#     result = analyze_image(save_path)

#     return render_template("result.html", result=result)

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)


# from flask import Flask, render_template, request
# import os
# from inference import analyze_image
# from video_inference import analyze_video

# app = Flask(__name__)

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# VIDEO_EXTS = {".mp4", ".avi"}


# @app.route("/")
# def home():
#     return render_template("index.html")


# @app.route("/analyze", methods=["POST"])
# def analyze():
#     file = request.files["media"]
#     save_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(save_path)

#     ext = os.path.splitext(file.filename)[1].lower()

#     if ext in VIDEO_EXTS:
#         result = analyze_video(save_path)
#         media_type = "video"
#     elif ext in IMAGE_EXTS:
#         result = analyze_image(save_path)
#         media_type = "image"
#     else:
#         return f"Unsupported file type: {ext}", 400

#     return render_template("result.html", result=result, media_type=media_type)


# if __name__ == "__main__":
#     app.run(debug=True, port=5000)



# from flask import Flask, render_template, request
# import os
# from inference import analyze_image, analyze_video

# app = Flask(__name__)

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# IMAGE_EXTS = {"jpg", "jpeg", "png"}
# VIDEO_EXTS = {"mp4", "avi"}

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/analyze", methods=["POST"])
# def analyze():
#     file = request.files["media"]
#     save_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(save_path)

#     ext = file.filename.rsplit(".", 1)[1].lower()

#     if ext in IMAGE_EXTS:
#         result = analyze_image(save_path)
#         media_type = "image"
#     elif ext in VIDEO_EXTS:
#         result = analyze_video(save_path)
#         media_type = "video"
#     else:
#         return "Unsupported file type.", 400

#     return render_template("result.html", result=result, media_type=media_type)

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)


# import datetime
# from flask import Flask, render_template, request
# import os
# from inference import analyze_image, analyze_video

# app = Flask(__name__)

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# IMAGE_EXTS = {"jpg", "jpeg", "png"}
# VIDEO_EXTS = {"mp4", "avi"}
# ALLOWED_EXTS = IMAGE_EXTS | VIDEO_EXTS

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/analyze", methods=["POST"])
# def analyze():
#     file = request.files["media"]

#     if file.filename == "" or "." not in file.filename:
#         return render_template("index.html", error="Please choose a valid file.")

#     ext = file.filename.rsplit(".", 1)[1].lower()

#     if ext not in ALLOWED_EXTS:
#         return render_template("index.html", error=f"Unsupported file type: .{ext}. Please upload JPG/PNG/JPEG or MP4/AVI.")

#     save_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(save_path)

#     upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     if ext in IMAGE_EXTS:
#         result = analyze_image(save_path)
#         media_type = "image"
#     else:
#         result = analyze_video(save_path)
#         media_type = "video"

#     # return render_template("result.html", result=result, media_type=media_type)
#     return render_template("result.html", result=result, media_type=media_type, upload_time=upload_time, filename=file.filename)

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)



# from flask import Flask, render_template, request, send_from_directory
# import os
# import datetime
# import uuid
# from inference import analyze_image, analyze_video
# from report import generate_report_pdf

# app = Flask(__name__)

# UPLOAD_FOLDER = "uploads"
# REPORTS_FOLDER = "reports"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(REPORTS_FOLDER, exist_ok=True)

# IMAGE_EXTS = {"jpg", "jpeg", "png"}
# VIDEO_EXTS = {"mp4", "avi"}
# ALLOWED_EXTS = IMAGE_EXTS | VIDEO_EXTS

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/analyze", methods=["POST"])
# def analyze():
#     file = request.files["media"]

#     if file.filename == "" or "." not in file.filename:
#         return render_template("index.html", error="Please choose a valid file.")

#     ext = file.filename.rsplit(".", 1)[1].lower()

#     if ext not in ALLOWED_EXTS:
#         return render_template("index.html", error=f"Unsupported file type: .{ext}. Please upload JPG/PNG/JPEG or MP4/AVI.")

#     save_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(save_path)

#     upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     if ext in IMAGE_EXTS:
#         result = analyze_image(save_path)
#         media_type = "image"
#     else:
#         result = analyze_video(save_path)
#         media_type = "video"

#     report_id = uuid.uuid4().hex[:8]
#     report_path = os.path.join(REPORTS_FOLDER, f"{report_id}.pdf")
#     generate_report_pdf(result, file.filename, upload_time, media_type, report_path)

#     return render_template(
#         "result.html",
#         result=result,
#         media_type=media_type,
#         upload_time=upload_time,
#         filename=file.filename,
#         report_id=report_id,
#     )

# @app.route("/download_report/<report_id>")
# def download_report(report_id):
#     return send_from_directory(REPORTS_FOLDER, f"{report_id}.pdf", as_attachment=True)

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)


from flask import Flask, render_template, request, send_from_directory
import os
import datetime
import uuid
from inference import analyze_image, analyze_video
from report import generate_report_pdf

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
REPORTS_FOLDER = "reports"
OUTPUTS_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

IMAGE_EXTS = {"jpg", "jpeg", "png"}
VIDEO_EXTS = {"mp4", "avi"}
ALLOWED_EXTS = IMAGE_EXTS | VIDEO_EXTS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["media"]

    if file.filename == "" or "." not in file.filename:
        return render_template("index.html", error="Please choose a valid file.")

    ext = file.filename.rsplit(".", 1)[1].lower()

    if ext not in ALLOWED_EXTS:
        return render_template("index.html", error=f"Unsupported file type: .{ext}. Please upload JPG/PNG/JPEG or MP4/AVI.")

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    job_id = uuid.uuid4().hex[:8]
    heatmap_path = os.path.join(OUTPUTS_FOLDER, f"{job_id}_heatmap.jpg")

    if ext in IMAGE_EXTS:
        result = analyze_image(save_path, heatmap_path)
        media_type = "image"
    else:
        result = analyze_video(save_path, heatmap_path)
        media_type = "video"

    report_path = os.path.join(REPORTS_FOLDER, f"{job_id}.pdf")
    generate_report_pdf(result, file.filename, upload_time, media_type, report_path)

    return render_template(
        "result.html",
        result=result,
        media_type=media_type,
        upload_time=upload_time,
        filename=file.filename,
        report_id=job_id,
    )

@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUTS_FOLDER, filename)

@app.route("/download_report/<report_id>")
def download_report(report_id):
    return send_from_directory(REPORTS_FOLDER, f"{report_id}.pdf", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port=5000)