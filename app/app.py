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

from flask import Flask, render_template, request
import os
from inference import analyze_image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["media"]
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    result = analyze_image(save_path)

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)