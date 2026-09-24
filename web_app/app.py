from flask import Flask, render_template, request, redirect
import os
import requests
#webitf
app = Flask(__name__)
RESULT_FOLDER = "static/results"
os.makedirs(RESULT_FOLDER, exist_ok=True)

PI_IP = "http://172.16.18.30:5000"  # Replace with your Pi's IP address

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["video"]
    filename = file.filename
    file.save(filename)

    # Send to Raspberry Pi
    with open(filename, "rb") as f:
        requests.post(f"{PI_IP}/process", files={"video": f})

    os.remove(filename)
    return redirect("/results")

@app.route("/results")
def results():
    files = ["gun_top.jpg", "mask_top.jpg"]
    downloaded = []

    for f in files:
        url = f"{PI_IP}/results/{f}"
        r = requests.get(url)
        if r.status_code == 200:
            with open(os.path.join(RESULT_FOLDER, f), "wb") as out:
                out.write(r.content)
            downloaded.append(f)

    # Download output video
    r = requests.get(f"{PI_IP}/video")
    with open(os.path.join(RESULT_FOLDER, "output_video.mp4"), "wb") as out:
        out.write(r.content)

    return render_template("results.html", images=downloaded)
    
if __name__ == "__main__":
    app.run(debug=True)