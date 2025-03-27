import os
import shutil
import glob
import requests
import cv2
import numpy as np
import time
from flask import Flask, request, render_template_string, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from deepface import DeepFace

# =============================================================================
# Configuration
# =============================================================================
class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    COMPARE_FOLDER = os.path.join(UPLOAD_FOLDER, "compare")
    TARGET_FOLDER = os.path.join(UPLOAD_FOLDER, "target")
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def create_directories():
    for folder in [Config.COMPARE_FOLDER, Config.TARGET_FOLDER]:
        os.makedirs(folder, exist_ok=True)
        for sub in ["extracted", "selected"]:
            os.makedirs(os.path.join(folder, sub), exist_ok=True)

create_directories()

# =============================================================================
# Helper to Recursively Convert NumPy Types for JSON Serialization
# =============================================================================
def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

# =============================================================================
# Helper to Extract Dominant Attribute (e.g. gender)
# =============================================================================
def get_dominant_attribute(result, attr, default="N/A"):
    value = result.get(attr, default)
    if isinstance(value, dict):
        return max(value.items(), key=lambda x: x[1])[0]
    return value

# =============================================================================
# Flask App Initialization and DeepFace Config Defaults
# =============================================================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER_COMPARE'] = Config.COMPARE_FOLDER
app.config['UPLOAD_FOLDER_TARGET'] = Config.TARGET_FOLDER
app.config['ANALYSIS_METHOD'] = "deepface"  # Only DeepFace is used

app.config["MODEL_NAME"] = "ArcFace"
app.config["DISTANCE_METRIC"] = "cosine"
app.config["DETECTOR_BACKEND"] = "retinaface"

# =============================================================================
# Utility Functions
# =============================================================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        else:
            app.logger.error("Download failed, status code: %s", response.status_code)
    except Exception as e:
        app.logger.error("Download error: %s", e)
    return False

# =============================================================================
# Face Extraction & Auto-Selection
# =============================================================================
def extract_and_save_faces(image_path, category, filename):
    if filename.lower().endswith('.png'):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            cv2.imwrite(image_path, img)

    detector_backend = app.config.get("DETECTOR_BACKEND", "retinaface")
    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend=detector_backend,
            enforce_detection=True,
            align=True
        )
    except Exception as e:
        app.logger.error("Extraction error for %s: %s", filename, e)
        faces = []
    
    base = Config.TARGET_FOLDER if category == "target" else Config.COMPARE_FOLDER
    extracted_folder = os.path.join(base, "extracted")
    selected_folder = os.path.join(base, "selected")
    
    if len(faces) == 0:
        if os.path.exists(image_path):
            os.remove(image_path)
        app.logger.info("No face detected in %s, file removed.", filename)
        return {"status": "none", "message": f"No se detectó cara en {filename}."}
    elif len(faces) == 1:
        face_img = faces[0]["face"]
        if face_img.dtype != np.uint8:
            face_img = (face_img * 255).astype(np.uint8) if face_img.max() <= 1 else face_img.astype(np.uint8)
        face_filename = os.path.splitext(filename)[0] + "_face1.jpg"
        face_path = os.path.join(extracted_folder, face_filename)
        cv2.imwrite(face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
        shutil.copyfile(face_path, os.path.join(selected_folder, filename))
        os.remove(image_path)
        return {"status": "single"}
    else:
        face_info_list = []
        for i, face in enumerate(faces):
            face_img = face["face"]
            if face_img.dtype != np.uint8:
                face_img = (face_img * 255).astype(np.uint8) if face_img.max() <= 1 else face_img.astype(np.uint8)
            face_filename = os.path.splitext(filename)[0] + f"_face{i+1}.jpg"
            face_path = os.path.join(extracted_folder, face_filename)
            cv2.imwrite(face_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            face_info_list.append({"face_index": str(i+1), "face_filename": face_filename})
        os.remove(image_path)
        return {"status": "multiple", "faces": face_info_list, "original": filename, "category": category}

# =============================================================================
# Endpoints for File Serving and Management
# =============================================================================
@app.route('/uploads/compare/<path:filename>')
def uploaded_compare(filename):
    return send_from_directory(Config.COMPARE_FOLDER, filename)

@app.route('/uploads/target/<path:filename>')
def uploaded_target(filename):
    return send_from_directory(Config.TARGET_FOLDER, filename)

@app.route('/uploads/compare/extracted/<path:filename>')
def uploaded_compare_extracted(filename):
    base = Config.COMPARE_FOLDER
    return send_from_directory(os.path.join(base, "extracted"), filename)

@app.route('/uploads/target/extracted/<path:filename>')
def uploaded_target_extracted(filename):
    base = Config.TARGET_FOLDER
    return send_from_directory(os.path.join(base, "extracted"), filename)

@app.route("/list/<category>")
def list_files(category):
    result = []
    if category == "target":
        selected_folder = os.path.join(Config.TARGET_FOLDER, "selected")
        url_prefix = "/uploads/target/selected/"
    elif category == "compare":
        selected_folder = os.path.join(Config.COMPARE_FOLDER, "selected")
        url_prefix = "/uploads/compare/selected/"
    else:
        return jsonify(files=[])
    for f in os.listdir(selected_folder):
        full_path = os.path.join(selected_folder, f)
        if os.path.isfile(full_path):
            result.append({"filename": f, "display_url": url_prefix + f})
    return jsonify(files=result)

# =============================================================================
# Delete Endpoint
# =============================================================================
@app.route("/delete/<category>/<filename>", methods=["POST"])
def delete_image(category, filename):
    if category == "target":
        base = Config.TARGET_FOLDER
    elif category == "compare":
        base = Config.COMPARE_FOLDER
    else:
        return jsonify({"status": "error", "message": "Categoría inválida"}), 400
    selected_path = os.path.join(base, "selected", filename)
    if os.path.exists(selected_path):
        os.remove(selected_path)
    extracted_folder = os.path.join(base, "extracted")
    pattern = os.path.join(extracted_folder, os.path.splitext(filename)[0] + "_face*.jpg")
    for f in glob.glob(pattern):
        os.remove(f)
    return jsonify({"status": "success", "message": "Imagen eliminada"})

def handle_upload(category, upload_folder):
    messages = []
    files = request.files.getlist('files[]')
    if files:
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                if not filename:
                    filename = f"uploaded_{int(time.time())}.jpg"
                save_path = os.path.join(upload_folder, filename)
                file.save(save_path)
                res = extract_and_save_faces(save_path, category, filename)
                if res["status"] == "single":
                    messages.append({"filename": filename, "message": f"{filename} subido correctamente."})
                elif res["status"] == "none":
                    messages.append({"filename": filename, "message": res.get("message", f"{filename} omitido (sin cara detectada).")})
                elif res["status"] == "multiple":
                    messages.append({
                        "filename": filename,
                        "message": f"Se detectaron múltiples caras en {filename}. Por favor, selecciona la(s) cara(s) deseada(s).",
                        "multiple": True,
                        "faces": res["faces"],
                        "category": res["category"],
                        "original": res["original"]
                    })
    image_url = request.form.get('image_url')
    if image_url:
        filename = secure_filename(image_url.split('/')[-1])
        if not filename:
            filename = f"downloaded_{int(time.time())}.jpg"
        save_path = os.path.join(upload_folder, filename)
        if download_image(image_url, save_path):
            res = extract_and_save_faces(save_path, category, filename)
            if res["status"] == "single":
                messages.append({"filename": filename, "message": f"{filename} subido correctamente."})
            elif res["status"] == "none":
                messages.append({"filename": filename, "message": res.get("message", f"{filename} omitido (sin cara detectada).")})
            elif res["status"] == "multiple":
                messages.append({
                    "filename": filename,
                    "message": f"Se detectaron múltiples caras en {filename}. Por favor, selecciona la(s) cara(s) deseada(s).",
                    "multiple": True,
                    "faces": res["faces"],
                    "category": res["category"],
                    "original": res["original"]
                })
        else:
            messages.append({"filename": filename, "message": f"Error al descargar la imagen desde la URL: {image_url}"})
    return messages

@app.route("/upload/compare", methods=['POST'])
def upload_compare():
    messages = handle_upload("compare", Config.COMPARE_FOLDER)
    return jsonify({"status": "success", "category": "compare", "messages": messages})

@app.route("/upload/target", methods=['POST'])
def upload_target():
    messages = handle_upload("target", Config.TARGET_FOLDER)
    return jsonify({"status": "success", "category": "target", "messages": messages})

# =============================================================================
# Endpoint for comparing a pair of images (individual analysis)
# =============================================================================
@app.route("/analyze_pair", methods=["GET"])
def analyze_pair():
    analysis_type = request.args.get("analysis_type", "verification")
    target = request.args.get("target")
    if analysis_type == "verification":
        compare = request.args.get("compare")
        if not target or not compare:
            return jsonify({"error": "Faltan parámetros"}), 400
    else:
        if not target:
            return jsonify({"error": "Falta el parámetro target"}), 400

    detector_backend = app.config.get("DETECTOR_BACKEND", "retinaface")
    model_name = app.config.get("MODEL_NAME", "ArcFace")
    distance_metric = app.config.get("DISTANCE_METRIC", "cosine")
    target_selected_folder = os.path.join(Config.TARGET_FOLDER, "selected")
    compare_selected_folder = os.path.join(Config.COMPARE_FOLDER, "selected")
    t_path = os.path.join(target_selected_folder, target) if os.path.exists(os.path.join(target_selected_folder, target)) else os.path.join(Config.TARGET_FOLDER, target)
    t_url = "/uploads/target/selected/" + target if os.path.exists(os.path.join(target_selected_folder, target)) else "/uploads/target/" + target

    if analysis_type == "verification":
        compare = request.args.get("compare")
        c_path = os.path.join(compare_selected_folder, compare) if os.path.exists(os.path.join(compare_selected_folder, compare)) else os.path.join(Config.COMPARE_FOLDER, compare)
        c_url = "/uploads/compare/selected/" + compare if os.path.exists(os.path.join(compare_selected_folder, compare)) else "/uploads/compare/" + compare
        try:
            try:
                threshold = float(request.args.get("threshold", 0.4))
            except ValueError:
                threshold = 0.4
            res = DeepFace.verify(
                img1_path=t_path,
                img2_path=c_path,
                enforce_detection=False,
                detector_backend=detector_backend,
                model_name=model_name,
                distance_metric=distance_metric,
                align=True
            )
            distance = res.get("distance", 0)
            verified = res.get("verified", False)
            if distance_metric == "cosine":
                similarity_percent = round(max(0, min(100, (1 - distance) * 100)), 2)
            else:
                similarity_percent = round(max(0, min(100, (1 - (distance / threshold)) * 100)), 2)
            return jsonify({
                "investigated": target,
                "comparison": compare,
                "investigated_url": t_url,
                "comparison_url": c_url,
                "verified": verified,
                "similarity": f"{similarity_percent} %"
            })
        except Exception as e:
            return jsonify({
                "investigated": target,
                "comparison": compare,
                "investigated_url": t_url,
                "comparison_url": c_url,
                "error": str(e)
            })
    elif analysis_type == "attributes":
        try:
            analysis_result = DeepFace.analyze(
                img_path=t_path,
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=False,
                detector_backend=detector_backend,
                align=True
            )
            if isinstance(analysis_result, list):
                if len(analysis_result) == 1:
                    analysis_result = analysis_result[0]
                else:
                    return jsonify({
                        "image": target,
                        "image_url": t_url,
                        "error": "Multiple faces detected in the selected image. Please re-select a single face."
                    }), 400
            analysis_result = make_serializable(analysis_result)
            return jsonify({
                "image": target,
                "image_url": t_url,
                "age": analysis_result.get("age", "N/A"),
                "gender": get_dominant_attribute(analysis_result, "gender", "N/A"),
                "race": analysis_result.get("dominant_race", "N/A"),
                "emotion": analysis_result.get("dominant_emotion", "N/A")
            })
        except Exception as e:
            return jsonify({
                "image": target,
                "image_url": t_url,
                "error": str(e)
            })
    else:
        return jsonify({"error": "Tipo de análisis inválido"}), 400

# =============================================================================
# DeepFace Configuration Endpoint
# =============================================================================
@app.route("/set_deepface_config", methods=["POST"])
def set_deepface_config():
    data = request.get_json() or {}
    model_name = data.get("model_name")
    distance_metric = data.get("distance_metric")
    detector_backend = data.get("detector_backend")
    if model_name:
        app.config["MODEL_NAME"] = model_name
    if distance_metric:
        app.config["DISTANCE_METRIC"] = distance_metric
    if detector_backend:
        app.config["DETECTOR_BACKEND"] = detector_backend
    return jsonify({
        "status": "success",
        "model_name": app.config.get("MODEL_NAME"),
        "distance_metric": app.config.get("DISTANCE_METRIC"),
        "detector_backend": app.config.get("DETECTOR_BACKEND")
    })

# =============================================================================
# Endpoint to select face(s) when multiple are detected
# =============================================================================
@app.route("/select_face", methods=["POST"])
def select_face():
    data = request.get_json()
    category = data.get("category")
    original = data.get("original")
    face_index = data.get("face_index")
    face_indices = data.get("face_indices")
    
    if category not in ["target", "compare"]:
        return jsonify({"status": "error", "message": "Categoría inválida"}), 400
    
    base = Config.TARGET_FOLDER if category == "target" else Config.COMPARE_FOLDER
    extracted_folder = os.path.join(base, "extracted")
    selected_folder = os.path.join(base, "selected")
    
    if face_indices:
        selected_faces = []
        for idx in face_indices:
            face_filename = os.path.splitext(original)[0] + f"_face{idx}.jpg"
            face_path = os.path.join(extracted_folder, face_filename)
            if os.path.exists(face_path):
                dest_filename = os.path.splitext(original)[0] + f"_face{idx}.jpg"
                shutil.copyfile(face_path, os.path.join(selected_folder, dest_filename))
                selected_faces.append(dest_filename)
            else:
                return jsonify({"status": "error", "message": f"Face image {face_filename} not found."}), 404
        return jsonify({"status": "success", "message": "Faces selected successfully!", "selected_faces": selected_faces})
    elif face_index:
        face_filename = os.path.splitext(original)[0] + f"_face{face_index}.jpg"
        face_path = os.path.join(extracted_folder, face_filename)
        if os.path.exists(face_path):
            shutil.copyfile(face_path, os.path.join(selected_folder, original))
            return jsonify({"status": "success", "message": "Face selected successfully!"})
        else:
            return jsonify({"status": "error", "message": "Face image not found."}), 404
    else:
        return jsonify({"status": "error", "message": "No face index provided."}), 400

# =============================================================================
# Front-End Template & Index Route (Enhanced UI/UX)
# =============================================================================
TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DeepFace Recognition Pipeline</title>
  <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/font-awesome@4.7.0/css/font-awesome.min.css" rel="stylesheet">
  <style>
    :root {
      --primary: #ff80ab;
      --secondary: #ff4081;
      --accent: #ffd1dc;
      --bg: #fff0f5;
      --text: #333;
      --white: #fff;
      --border-radius: 6px;
      --transition-speed: 0.3s;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { 
      font-family: 'Roboto', sans-serif; 
      background: var(--bg); 
      color: var(--text); 
      transition: background var(--transition-speed), color var(--transition-speed);
      line-height: 1.6;
      padding-bottom: 40px;
    }
    a { text-decoration: none; color: inherit; }
    header { 
      background: var(--primary); 
      padding: 25px; 
      text-align: center; 
      border-bottom: 2px solid var(--secondary);
      margin-bottom: 20px;
    }
    header h1 { font-size: 2rem; color: var(--white); margin-bottom: 5px; }
    header p { color: var(--white); }
    nav { 
      display: flex; 
      justify-content: center; 
      background: var(--accent);
      margin-bottom: 20px;
      flex-wrap: wrap;
    }
    nav a { 
      padding: 15px 30px; 
      transition: background var(--transition-speed), transform 0.2s;
      color: var(--text);
      margin: 5px;
      border-radius: var(--border-radius);
    }
    nav a:hover, nav a.active { 
      background: var(--primary); 
      color: var(--white); 
      transform: translateY(-2px); 
    }
    nav a:focus { outline: 2px dashed var(--secondary); }
    .container { 
      max-width: 1200px; 
      margin: 0 auto; 
      padding: 20px; 
      background: var(--white); 
      border-radius: 8px; 
      box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .section { 
      display: none; 
      animation: fadeIn 0.5s ease-in-out;
      padding-bottom: 20px;
    }
    .section.active { display: block; }
    form { text-align: center; margin-bottom: 20px; }
    form input[type="file"],
    form input[type="text"],
    form input[type="number"],
    form select {
      padding: 10px; 
      margin: 10px 5px; 
      border: 1px solid #ccc; 
      border-radius: 4px;
      width: 80%; 
      max-width: 400px; 
      background: var(--white); 
      color: var(--text); 
      transition: border-color var(--transition-speed);
    }
    form input:focus, form select:focus { border-color: var(--primary); outline: none; }
    #threshold-input {
      border: 2px solid var(--primary);
      border-radius: 4px;
      padding: 10px;
      margin: 10px;
      width: 100px;
    }
    select {
      background-color: var(--white);
      border: 1px solid var(--primary);
      color: var(--text);
      padding: 10px;
      border-radius: 4px;
      appearance: none;
      background-image: url('data:image/svg+xml;charset=US-ASCII,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="gray" viewBox="0 0 16 16"><path d="M4.646 6.646a.5.5 0 01.708 0L8 9.293l2.646-2.647a.5.5 0 11.708.708l-3 3a.5.5 0 01-.708 0l-3-3a.5.5 0 010-.708z"/></svg>');
      background-repeat: no-repeat;
      background-position: right 10px center;
      background-size: 16px 16px;
    }
    .drop-zone { 
      border: 2px dashed var(--primary); 
      border-radius: 4px; 
      padding: 20px; 
      text-align: center; 
      margin: 10px auto; 
      max-width: 400px; 
      cursor: pointer; 
      transition: background var(--transition-speed);
    }
    .drop-zone.dragover { background-color: #ffe6f0; }
    button, form button {
      padding: 12px 24px;
      background: linear-gradient(145deg, var(--primary), var(--secondary));
      border: none; 
      color: var(--white); 
      border-radius: var(--border-radius); 
      cursor: pointer;
      transition: transform 0.2s, box-shadow 0.2s;
      margin: 5px;
    }
    button:hover, form button:hover { 
      transform: translateY(-3px); 
      box-shadow: 0 4px 12px rgba(0,0,0,0.2); 
    }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .card-grid { 
      display: grid; 
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); 
      gap: 20px; 
    }
    .card { 
      background: var(--white); 
      border: 1px solid #ffc0cb; 
      border-radius: var(--border-radius); 
      padding: 15px; 
      text-align: center; 
      transition: transform 0.3s, box-shadow 0.3s; 
      cursor: pointer; 
    }
    .card:hover { 
      transform: translateY(-5px); 
      box-shadow: 0 4px 12px rgba(0,0,0,0.15); 
    }
    .card img { 
      width: 150px; 
      height: 150px; 
      object-fit: cover; 
      border-radius: 4px; 
      transition: opacity 0.3s; 
    }
    .card .filename {
      margin-top: 8px; 
      font-size: 0.9em;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: 100%;
    }
    /* Enhanced Analysis Result Card Layout */
    .result-card {
      background: var(--white);
      border: 1px solid #ffc0cb;
      border-radius: var(--border-radius);
      padding: 15px;
      margin: 10px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 15px;
      transition: transform 0.3s, box-shadow 0.3s, opacity 0.5s;
    }
    .result-card:hover {
      transform: translateY(-5px);
      box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .result-card .result-images {
      flex-shrink: 0;
      display: flex;
      gap: 10px;
      margin-bottom: 0;
    }
    .result-card .result-thumb {
      width: 120px;
      height: 120px;
      object-fit: cover;
      border-radius: 4px;
    }
    .result-card .result-info {
      flex-grow: 1;
      text-align: left;
      font-size: 0.95em;
      line-height: 1.4;
    }
    .result-card .result-info h3 {
      margin-bottom: 8px;
      font-size: 1.1em;
    }
    .progress-container {
      background: #eee;
      border-radius: 4px;
      overflow: hidden;
      margin-top: 5px;
      height: 8px;
    }
    .progress-bar {
      height: 8px;
      background: var(--primary);
      border-radius: 4px;
      transition: width 0.3s ease;
    }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    #toast {
      position: fixed; 
      bottom: 20px; 
      right: 20px;
      background: #333; 
      color: var(--white); 
      padding: 15px 20px;
      border-radius: 4px; 
      opacity: 0; 
      transition: opacity 0.5s ease-in-out;
      z-index: 1000;
    }
    #toast.show { opacity: 1; }
    #modal, #faceSelectionModal, #deleteConfirmationModal {
      display: none; 
      position: fixed; 
      z-index: 2000; 
      left: 0; 
      top: 0;
      width: 100%; 
      height: 100%; 
      overflow: auto;
      background-color: rgba(0,0,0,0.8);
      backdrop-filter: blur(4px);
    }
    #modal-content { 
      margin: 10% auto; 
      display: block; 
      max-width: 90%;
      animation: modalFadeIn 0.3s ease-out;
    }
    #modal-close { 
      position: absolute; 
      top: 20px; 
      right: 35px; 
      color: var(--white); 
      font-size: 40px; 
      font-weight: bold; 
      cursor: pointer; 
    }
    #faceSelectionModal .modal-content,
    #deleteConfirmationModal .modal-content {
      background: var(--white); 
      margin: 10% auto; 
      padding: 20px; 
      border-radius: 8px; 
      max-width: 500px; 
      text-align: center; 
      position: relative;
      animation: modalFadeIn 0.3s ease-out;
    }
    #faceSelectionModal .close, #deleteConfirmationModal .close {
      position: absolute; 
      top: 10px; 
      right: 15px; 
      color: #aaa; 
      font-size: 28px; 
      font-weight: bold; 
      cursor: pointer;
    }
    #faceSelectionModal .close:hover, #deleteConfirmationModal .close:hover { color: #000; }
    #uploadSpinner {
      position: fixed; 
      top: 0; left: 0; 
      width: 100%; 
      height: 100%;
      background: rgba(255,255,255,0.7);
      display: none; 
      justify-content: center; 
      align-items: center; 
      z-index: 1500;
    }
    .la-ball-atom, .la-ball-atom > div { position: relative; box-sizing: border-box; }
    .la-ball-atom { display: inline-block; font-size: 0; color: var(--secondary); width: 32px; height: 32px; }
    .la-2x { transform: scale(2); }
    .la-ball-atom > div { display: inline-block; background-color: currentColor; border: 0 solid currentColor; }
    .la-ball-atom > div:nth-child(1) {
      position: absolute; top: 50%; left: 50%; z-index: 1; width: 60%; height: 60%;
      background: #ffc0cb; border-radius: 100%; transform: translate(-50%, -50%);
      animation: ball-atom-shrink 4.5s infinite linear;
    }
    .la-ball-atom > div:not(:nth-child(1)) {
      position: absolute; left: 0; z-index: 0; width: 100%; height: 100%;
      background: none; animation: ball-atom-zindex 1.5s 0s infinite steps(2, end);
    }
    .la-ball-atom > div:not(:nth-child(1)):before {
      position: absolute; top: 0; left: 0; width: 10px; height: 10px;
      margin-top: -5px; margin-left: -5px; content: ""; background: currentColor;
      border-radius: 50%; opacity: .75;
      animation: ball-atom-position 1.5s 0s infinite ease, ball-atom-size 1.5s 0s infinite ease;
    }
    .la-ball-atom > div:nth-child(2) { animation-delay: .75s; }
    .la-ball-atom > div:nth-child(2):before { animation-delay: 0s, -1.125s; }
    .la-ball-atom > div:nth-child(3) { transform: rotate(120deg); animation-delay: -.25s; }
    .la-ball-atom > div:nth-child(3):before { animation-delay: -1s, -.75s; }
    .la-ball-atom > div:nth-child(4) { transform: rotate(240deg); animation-delay: .25s; }
    .la-ball-atom > div:nth-child(4):before { animation-delay: -.5s, -.125s; }
    @keyframes ball-atom-position { 50% { top: 100%; left: 100%; } }
    @keyframes ball-atom-size { 50% { transform: scale(.5, .5); } }
    @keyframes ball-atom-zindex { 50% { z-index: 10; } }
    @keyframes ball-atom-shrink { 50% { transform: translate(-50%, -50%) scale(.8, .8); } }
    @keyframes modalFadeIn {
      from { transform: translateY(-20px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }
    @media (max-width: 600px) {
      .container { margin: 10px; padding: 10px; }
      nav { flex-direction: column; }
      .card-grid { grid-template-columns: 1fr; }
      form input[type="file"], form input[type="text"], form input[type="number"], form select { width: 100%; }
      .result-card { flex-direction: column; text-align: center; }
      .result-card .result-info { text-align: center; }
    }
  </style>
</head>
<body>
  <header>
    <h1><i class="fa fa-user" aria-hidden="true"></i> DeepFace UI</h1>
    <p>State-of-the-art facial recognition &amp; analysis</p>
  </header>
  <nav aria-label="Main Navigation">
    <a href="#" class="nav-link active" data-target="investigated" aria-label="Investigated Images">Investigated</a>
    <a href="#" class="nav-link" data-target="database" aria-label="Comparison Database">Database</a>
    <a href="#" class="nav-link" data-target="analysis" aria-label="Analysis">Analysis</a>
    <a href="#" class="nav-link" data-target="settings" aria-label="Settings">Settings</a>
  </nav>
  <div class="container">
    <!-- Investigated Images Section -->
    <section id="investigated" class="section active" aria-labelledby="investigated-heading">
      <h2 id="investigated-heading">Investigated Images</h2>
      <form id="upload-target-form" action="/upload/target" method="post" enctype="multipart/form-data" aria-label="Upload Investigated Images">
        <div class="drop-zone" id="drop-zone-target" tabindex="0">Drag &amp; drop images here or click to select</div>
        <input type="file" name="files[]" id="target-files" multiple accept="image/*" style="display:none;">
        <input type="text" name="image_url" placeholder="Or enter image URL" aria-label="Image URL">
        <button type="submit" aria-label="Upload Investigated Images">Upload Investigated Images</button>
      </form>
      <div class="card-grid" id="target-grid"></div>
    </section>
    
    <!-- Comparison Database Section -->
    <section id="database" class="section" aria-labelledby="database-heading">
      <h2 id="database-heading">Comparison Database</h2>
      <form id="upload-compare-form" action="/upload/compare" method="post" enctype="multipart/form-data" aria-label="Upload Comparison Images">
        <div class="drop-zone" id="drop-zone-compare" tabindex="0">Drag &amp; drop images here or click to select</div>
        <input type="file" name="files[]" id="compare-files" multiple accept="image/*" style="display:none;">
        <input type="text" name="image_url" placeholder="Or enter image URL" aria-label="Image URL">
        <button type="submit" aria-label="Upload Comparison Images">Upload Comparison Images</button>
      </form>
      <div class="card-grid" id="compare-grid"></div>
    </section>
    
    <!-- Analysis Section -->
    <section id="analysis" class="section" aria-labelledby="analysis-heading">
      <h2 id="analysis-heading">Analysis</h2>
      <div class="analysis-config">
        <label for="analysis-type">Analysis Type:</label>
        <select id="analysis-type" aria-label="Select Analysis Type">
          <option value="verification">Verification</option>
          <option value="attributes">Attributes</option>
        </select>
        <span id="threshold-container">
          <label for="threshold-input">Similarity Threshold:</label>
          <input type="number" id="threshold-input" value="0.4" step="0.01" min="0" aria-label="Similarity Threshold">
          <small>(Lower yields higher similarity %)</small>
        </span>
      </div>
      <button id="run-analysis" aria-label="Run Analysis">Run Analysis</button>
      <div id="spinner" style="display:none; text-align:center; margin-top:20px;">
        <div class="la-ball-atom la-2x"><div></div><div></div><div></div><div></div></div>
      </div>
      <div id="analysis-results"></div>
    </section>
    
    <!-- Settings Section -->
    <section id="settings" class="section" aria-labelledby="settings-heading">
      <h2 id="settings-heading">DeepFace Configuration</h2>
      <div class="config-form">
        <label for="model-select">Model:</label>
        <select id="model-select" aria-label="Select Model">
          <option value="VGG-Face">VGG-Face</option>
          <option value="Facenet">Facenet</option>
          <option value="Facenet512">Facenet512</option>
          <option value="OpenFace">OpenFace</option>
          <option value="DeepFace">DeepFace</option>
          <option value="DeepID">DeepID</option>
          <option value="ArcFace" selected>ArcFace</option>
          <option value="Dlib">Dlib</option>
          <option value="SFace">SFace</option>
          <option value="GhostFaceNet">GhostFaceNet</option>
          <option value="Buffalo_L">Buffalo_L</option>
        </select>
        <label for="metric-select">Distance Metric:</label>
        <select id="metric-select" aria-label="Select Distance Metric">
          <option value="cosine" selected>Cosine</option>
          <option value="euclidean">Euclidean</option>
          <option value="euclidean_l2">Euclidean L2</option>
        </select>
        <label for="detector-select">Detector Backend:</label>
        <select id="detector-select" aria-label="Select Detector Backend">
          <option value="opencv">OpenCV</option>
          <option value="ssd">SSD</option>
          <option value="dlib">Dlib</option>
          <option value="mtcnn">MTCNN</option>
          <option value="retinaface" selected>RetinaFace</option>
        </select>
        <button id="set-config" aria-label="Set Configuration">Set Configuration</button>
        <div class="feedback" id="config-feedback"></div>
      </div>
      <div class="feedback">Currently using DeepFace for all recognition tasks.</div>
    </section>
  </div>
  
  <!-- Toast Notification -->
  <div id="toast" role="alert" aria-live="assertive"></div>
  
  <!-- Modal for Image Preview -->
  <div id="modal" role="dialog" aria-modal="true">
    <span id="modal-close" aria-label="Close Modal">&times;</span>
    <img id="modal-content" alt="Image Preview">
  </div>
  
  <!-- Modal for Face Selection -->
  <div id="faceSelectionModal" role="dialog" aria-modal="true">
    <div class="modal-content">
      <span id="faceSelectionClose" class="close" aria-label="Close Face Selection">&times;</span>
      <h3>Select Face(s) for Upload</h3>
      <div id="faceOptions" style="display:flex; flex-wrap: wrap; gap: 10px; justify-content:center;"></div>
      <button id="confirmFaceSelection">Confirm Selection</button>
    </div>
  </div>
  
  <!-- Modal for Delete Confirmation -->
  <div id="deleteConfirmationModal" role="dialog" aria-modal="true">
    <div class="modal-content">
      <span id="deleteModalClose" class="close" aria-label="Close Delete Confirmation">&times;</span>
      <p>¿Estás seguro de borrar esta imagen?</p>
      <button id="deleteConfirmButton">Confirmar</button>
      <button id="deleteCancelButton">Cancelar</button>
    </div>
  </div>
  
  <!-- Upload Spinner Overlay -->
  <div id="uploadSpinner">
    <div class="la-ball-atom la-2x"><div></div><div></div><div></div><div></div></div>
  </div>
  
  <script>
    // Toast notifications
    function showToast(message, type='success') {
      const toast = document.getElementById('toast');
      toast.textContent = message;
      toast.style.background = type === 'error' ? '#cc0000' : '#333';
      toast.classList.add('show');
      setTimeout(() => { toast.classList.remove('show'); }, 3000);
    }
    
    // Delete confirmation variables
    let currentDeleteCategory = null, currentDeleteFilename = null, currentDeleteCardId = null;
    
    function showDeleteConfirmation(category, filename, cardId) {
      currentDeleteCategory = category;
      currentDeleteFilename = filename;
      currentDeleteCardId = cardId;
      document.getElementById('deleteConfirmationModal').style.display = "block";
    }
    
    function closeDeleteModal() {
      document.getElementById('deleteConfirmationModal').style.display = "none";
      currentDeleteCategory = currentDeleteFilename = currentDeleteCardId = null;
    }
    
    function deleteImageConfirmed() {
      fetch('/delete/' + currentDeleteCategory + '/' + currentDeleteFilename, { method: 'POST' })
      .then(response => response.json())
      .then(data => {
        if(data.status === "success") {
          document.getElementById(currentDeleteCardId).remove();
          showToast("Imagen eliminada exitosamente!");
        } else {
          showToast(data.message, 'error');
        }
        closeDeleteModal();
      })
      .catch(error => {
        console.error("Delete error:", error);
        closeDeleteModal();
      });
    }
    
    document.getElementById('deleteConfirmButton').addEventListener('click', deleteImageConfirmed);
    document.getElementById('deleteCancelButton').addEventListener('click', closeDeleteModal);
    document.getElementById('deleteModalClose').addEventListener('click', closeDeleteModal);
    
    // Navigation without page reload
    document.querySelectorAll('.nav-link').forEach(link => {
      link.addEventListener('click', function(e) {
        e.preventDefault();
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        document.querySelectorAll('.section').forEach(sec => sec.classList.remove('active'));
        this.classList.add('active');
        document.getElementById(this.getAttribute('data-target')).classList.add('active');
        if(this.getAttribute('data-target') === 'investigated'){
          loadImages('target', 'target-grid');
        } else if(this.getAttribute('data-target') === 'database'){
          loadImages('compare', 'compare-grid');
        }
      });
    });
    
    // Load images into grid
    function loadImages(category, gridId) {
      fetch('/list/' + category)
      .then(response => response.json())
      .then(data => {
        let gridHtml = '';
        data.files.forEach(file => {
          let cardId = category + '-' + file.filename;
          gridHtml += `
            <div class="card" id="${cardId}" tabindex="0" role="button" aria-label="Preview ${file.filename}" title="${file.filename}">
              <img src="${file.display_url}" alt="${file.filename}" class="preview-image">
              <div class="filename" title="${file.filename}">${file.filename}</div>
              <button class="delete" onclick="showDeleteConfirmation('${category}', '${file.filename}', '${cardId}')" aria-label="Delete ${file.filename}">
                <i class="fa fa-trash" aria-hidden="true"></i> Delete
              </button>
            </div>
          `;
        });
        document.getElementById(gridId).innerHTML = gridHtml;
        document.querySelectorAll('.card .preview-image').forEach(img => {
          img.addEventListener('click', function() {
            openModal(this.src, this.alt);
          });
        });
      });
    }
    
    // Analysis execution
    document.getElementById('run-analysis').addEventListener('click', function() {
      const analysisType = document.getElementById('analysis-type').value;
      const resultsContainer = document.getElementById('analysis-results');
      resultsContainer.innerHTML = "";
      document.getElementById('spinner').style.display = "block";
      document.getElementById('run-analysis').disabled = true;
      
      if(analysisType === "verification") {
        Promise.all([
          fetch('/list/target').then(res => res.json()),
          fetch('/list/compare').then(res => res.json())
        ]).then(([targetData, compareData]) => {
          const targets = targetData.files.map(f => f.filename);
          const compares = compareData.files.map(f => f.filename);
          let pairs = [];
          targets.forEach(t => {
            compares.forEach(c => {
              pairs.push({target: t, compare: c});
            });
          });
          function processPair(index) {
            if(index >= pairs.length) {
              document.getElementById('spinner').style.display = "none";
              document.getElementById('run-analysis').disabled = false;
              return;
            }
            const pair = pairs[index];
            const threshold = document.getElementById('threshold-input').value;
            const url = `/analyze_pair?analysis_type=verification&target=${encodeURIComponent(pair.target)}&compare=${encodeURIComponent(pair.compare)}&threshold=${threshold}`;
            fetch(url)
            .then(res => res.json())
            .then(result => {
              const card = document.createElement('div');
              card.classList.add('result-card');
              if(result.error) {
                card.innerHTML = `<div class="result-info"><span style="color:red;">${result.error}</span></div>`;
              } else {
                // Construct progress bar based on similarity percentage
                const similarityValue = result.similarity.replace(' %','');
                card.innerHTML = `<div class="result-images">
                    <img src="${result.investigated_url}" alt="${result.investigated}" class="result-thumb">
                    <img src="${result.comparison_url}" alt="${result.comparison}" class="result-thumb">
                  </div>
                  <div class="result-info">
                    <h3>${ result.verified ? '<i class="fa fa-check-circle" style="color: #4CAF50;"></i> Same Person' : '<i class="fa fa-times-circle" style="color: #F44336;"></i> Different Person' }</h3>
                    <div>Similarity: ${result.similarity}</div>
                    <div class="progress-container">
                      <div class="progress-bar" style="width: ${similarityValue}%;"></div>
                    </div>
                  </div>`;
              }
              card.style.opacity = 0;
              resultsContainer.appendChild(card);
              setTimeout(() => { card.style.opacity = 1; }, 50);
              setTimeout(() => processPair(index+1), 200);
            })
            .catch(error => {
              console.error("Error processing pair:", error);
              processPair(index+1);
            });
          }
          processPair(0);
        }).catch(error => {
          console.error("Error fetching image lists:", error);
          document.getElementById('spinner').style.display = "none";
          document.getElementById('run-analysis').disabled = false;
          showToast("Error al obtener la lista de imágenes.", "error");
        });
      } else if (analysisType === "attributes") {
        fetch('/list/target')
        .then(res => res.json())
        .then(data => {
          const targets = data.files.map(f => f.filename);
          function processAttribute(index) {
            if(index >= targets.length) {
              document.getElementById('spinner').style.display = "none";
              document.getElementById('run-analysis').disabled = false;
              return;
            }
            const target = targets[index];
            const url = `/analyze_pair?analysis_type=attributes&target=${encodeURIComponent(target)}`;
            fetch(url)
            .then(res => res.json())
            .then(result => {
              const card = document.createElement('div');
              card.classList.add('result-card');
              if(result.error) {
                card.innerHTML = `<div class="result-info"><span style="color:red;">${result.error}</span></div>`;
              } else {
                card.innerHTML = `<div class="result-images">
                    <img src="${result.image_url}" alt="${result.image}" class="result-thumb">
                  </div>
                  <div class="result-info">
                    <div><strong>${result.image}</strong></div>
                    <div>Age: ${result.age}</div>
                    <div>Gender: ${result.gender}</div>
                    <div>Race: ${result.race}</div>
                    <div>Emotion: ${result.emotion}</div>
                  </div>`;
              }
              card.style.opacity = 0;
              resultsContainer.appendChild(card);
              setTimeout(() => { card.style.opacity = 1; }, 50);
              setTimeout(() => processAttribute(index+1), 200);
            })
            .catch(error => {
              console.error("Error processing attribute for", target, error);
              processAttribute(index+1);
            });
          }
          processAttribute(0);
        }).catch(error => {
          console.error("Error fetching target image list:", error);
          document.getElementById('spinner').style.display = "none";
          document.getElementById('run-analysis').disabled = false;
          showToast("Error al obtener la lista de imágenes.", "error");
        });
      }
    });
    
    // Update DeepFace configuration
    document.getElementById('set-config').addEventListener('click', function() {
      const model = document.getElementById('model-select').value;
      const metric = document.getElementById('metric-select').value;
      const detector = document.getElementById('detector-select').value;
      const payload = { model_name: model, distance_metric: metric, detector_backend: detector };
      fetch('/set_deepface_config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      .then(response => response.json())
      .then(data => {
        const configFeedback = document.getElementById('config-feedback');
        configFeedback.textContent = "Configuration updated successfully!";
        showToast("Configuration updated successfully!");
        setTimeout(() => { configFeedback.textContent = ""; }, 3000);
      });
    });
    
    // Auto-upload from file explorer
    document.getElementById('target-files').addEventListener('change', function() {
      if(this.files.length > 0) {
        document.getElementById('upload-target-form').dispatchEvent(new Event('submit', {cancelable: true}));
      }
    });
    document.getElementById('compare-files').addEventListener('change', function() {
      if(this.files.length > 0) {
        document.getElementById('upload-compare-form').dispatchEvent(new Event('submit', {cancelable: true}));
      }
    });
    
    // AJAX upload for Investigated Images
    document.getElementById('upload-target-form').addEventListener('submit', function(e) {
      e.preventDefault();
      const startTime = Date.now();
      const formData = new FormData(this);
      document.getElementById('uploadSpinner').style.display = "flex";
      fetch(this.action, { method: 'POST', body: formData })
      .then(response => response.json())
      .then(data => {
         const elapsed = Date.now() - startTime;
         const delay = Math.max(500 - elapsed, 0);
         setTimeout(() => {
           document.getElementById('uploadSpinner').style.display = "none";
           data.messages.forEach(msg => {
             if(msg.multiple){
               showFaceSelectionModal(msg);
             } else {
               showToast(msg.message, msg.message.toLowerCase().includes("omitido") ? "error" : "success");
             }
           });
           loadImages('target', 'target-grid');
           this.reset();
         }, delay);
      })
      .catch(error => {
        console.error("Upload error (target):", error);
        document.getElementById('uploadSpinner').style.display = "none";
        showToast("Error al subir las imágenes.", "error");
      });
    });
    
    // AJAX upload for Comparison Images
    document.getElementById('upload-compare-form').addEventListener('submit', function(e) {
      e.preventDefault();
      const startTime = Date.now();
      const formData = new FormData(this);
      document.getElementById('uploadSpinner').style.display = "flex";
      fetch(this.action, { method: 'POST', body: formData })
      .then(response => response.json())
      .then(data => {
         const elapsed = Date.now() - startTime;
         const delay = Math.max(500 - elapsed, 0);
         setTimeout(() => {
           document.getElementById('uploadSpinner').style.display = "none";
           data.messages.forEach(msg => {
             if(msg.multiple){
               showFaceSelectionModal(msg);
             } else {
               showToast(msg.message, msg.message.toLowerCase().includes("omitido") ? "error" : "success");
             }
           });
           loadImages('compare', 'compare-grid');
           this.reset();
         }, delay);
      })
      .catch(error => {
        console.error("Upload error (compare):", error);
        document.getElementById('uploadSpinner').style.display = "none";
        showToast("Error al subir las imágenes.", "error");
      });
    });
    
    // Drag-and-drop for Target Upload
    const dropZoneTarget = document.getElementById('drop-zone-target');
    const targetFilesInput = document.getElementById('target-files');
    dropZoneTarget.addEventListener('click', () => targetFilesInput.click());
    dropZoneTarget.addEventListener('dragover', (e) => { e.preventDefault(); dropZoneTarget.classList.add('dragover'); });
    dropZoneTarget.addEventListener('dragleave', () => { dropZoneTarget.classList.remove('dragover'); });
    dropZoneTarget.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZoneTarget.classList.remove('dragover');
      targetFilesInput.files = e.dataTransfer.files;
    });
    
    // Drag-and-drop for Compare Upload
    const dropZoneCompare = document.getElementById('drop-zone-compare');
    const compareFilesInput = document.getElementById('compare-files');
    dropZoneCompare.addEventListener('click', () => compareFilesInput.click());
    dropZoneCompare.addEventListener('dragover', (e) => { e.preventDefault(); dropZoneCompare.classList.add('dragover'); });
    dropZoneCompare.addEventListener('dragleave', () => { dropZoneCompare.classList.remove('dragover'); });
    dropZoneCompare.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZoneCompare.classList.remove('dragover');
      compareFilesInput.files = e.dataTransfer.files;
    });
    
    // Modal for image preview
    function openModal(src, alt) {
      const modal = document.getElementById('modal');
      const modalContent = document.getElementById('modal-content');
      modal.style.display = "block";
      modalContent.src = src;
      modalContent.alt = alt;
    }
    document.getElementById('modal-close').addEventListener('click', () => {
      document.getElementById('modal').style.display = "none";
    });
    window.addEventListener('click', (e) => {
      const modal = document.getElementById('modal');
      if (e.target === modal) { modal.style.display = "none"; }
    });
    
    // Modal for Face Selection with multi-select support
    let currentMultiple = null;
    function showFaceSelectionModal(data) {
      currentMultiple = data;
      currentMultiple.selectedFaces = [];
      const faceOptions = document.getElementById('faceOptions');
      faceOptions.innerHTML = "";
      let baseUrl = data.category === 'target' ? '/uploads/target/extracted/' : '/uploads/compare/extracted/';
      data.faces.forEach(face => {
        let img = document.createElement('img');
        img.src = baseUrl + face.face_filename;
        img.alt = "Face " + face.face_index;
        img.dataset.faceIndex = face.face_index;
        img.style.width = "100px";
        img.style.height = "100px";
        img.style.objectFit = "cover";
        img.style.cursor = "pointer";
        img.style.border = "2px solid transparent";
        img.addEventListener('click', function() {
          let faceIndex = this.dataset.faceIndex;
          if(currentMultiple.selectedFaces.includes(faceIndex)){
              currentMultiple.selectedFaces = currentMultiple.selectedFaces.filter(idx => idx !== faceIndex);
              this.style.border = "2px solid transparent";
          } else {
              currentMultiple.selectedFaces.push(faceIndex);
              this.style.border = "2px solid var(--secondary)";
          }
        });
        faceOptions.appendChild(img);
      });
      document.getElementById('faceSelectionModal').style.display = "block";
    }
    document.getElementById('faceSelectionClose').addEventListener('click', () => {
      document.getElementById('faceSelectionModal').style.display = "none";
      currentMultiple = null;
    });
    document.getElementById('confirmFaceSelection').addEventListener('click', () => {
      if(currentMultiple && currentMultiple.selectedFaces && currentMultiple.selectedFaces.length > 0){
        const category = currentMultiple.category;
        const payload = {
          category: category,
          original: currentMultiple.original,
          face_indices: currentMultiple.selectedFaces
        };
        fetch('/select_face', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        })
        .then(response => response.json())
        .then(data => {
          showToast(data.message, data.status === "error" ? "error" : "success");
          let cat = category === 'target' ? 'target' : 'compare';
          loadImages(cat, cat + '-grid');
          document.getElementById('faceSelectionModal').style.display = "none";
          currentMultiple = null;
        })
        .catch(error => {
          console.error("Error en selección de cara:", error);
          showToast("Error en la selección de cara.", "error");
        });
      } else {
        showToast("Please select at least one face before confirming.", "error");
      }
    });
    
    // Initial image load
    loadImages('target', 'target-grid');
    loadImages('compare', 'compare-grid');
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(TEMPLATE)

# =============================================================================
# Run the App
# =============================================================================
if __name__ == "__main__":
    app.run(debug=True)
