from flask import Flask, render_template, request, session, redirect, url_for
import os
import sys
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import random
import pandas as pd
from inference import EyeDiseasePredictor

app = Flask(__name__,
            static_folder=os.path.join(os.path.dirname(__file__), 'static'),
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Path to the trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                          'Trained_Models', 'ODIR5K-Ensemble-Classification', 'ODIR5K_final.keras')

# Initialize the predictor
predictor = EyeDiseasePredictor(MODEL_PATH)

LABEL_MAP = {
    'N': {'en': 'Normal', 'vi': 'Bình thường'},
    'D': {'en': 'Diabetic Retinopathy', 'vi': 'Bệnh võng mạc đái tháo đường'},
    'G': {'en': 'Glaucoma', 'vi': 'Bệnh Glôcôm'},
    'C': {'en': 'Cataract', 'vi': 'Đục thủy tinh thể'},
    'A': {'en': 'Age-related Macular Degeneration', 'vi': 'Thoái hóa điểm vàng'},
    'H': {'en': 'Hypertensive Retinopathy', 'vi': 'Bệnh võng mạc cao huyết áp'},
    'M': {'en': 'Pathological Myopia', 'vi': 'Cận thị bệnh lý'},
    'O': {'en': 'Other Diseases', 'vi': 'Các bệnh lý khác'}
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.context_processor
def inject_globals():
    return {'now': datetime.now(), 'LABEL_MAP': LABEL_MAP}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', readImg = '0')

    if request.method == 'POST':
        if 'filename' not in request.files:
            return redirect(request.url)

        file = request.files['filename']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Perform prediction
            prediction = predictor.predict(filepath)

            # Format results for template
            mcc_label = prediction['mcc_label']
            mcc_name_en = LABEL_MAP[mcc_label]['en']
            mcc_name_vi = LABEL_MAP[mcc_label]['vi']
            mcc_prob = round(prediction['mcc_prob'] * 100, 1)

            # English and Vietnamese feedback for TTS
            voice_en = f"Detected {mcc_name_en} with {mcc_prob}% confidence."
            voice_vi = f"Phát hiện {mcc_name_vi} với độ tin cậy {mcc_prob}%."

            # Multi-label results
            mlc_results = []
            for lab, prob in prediction['all_mlc_probs'].items():
                mlc_results.append({
                    'code': lab,
                    'name': LABEL_MAP[lab]['en'],
                    'prob': round(prob * 100, 1)
                })

            # Sort by probability
            mlc_results = sorted(mlc_results, key=lambda x: x['prob'], reverse=True)

            return render_template('index.html',
                                 readImg = '1',
                                 img_url = f'uploads/{filename}',
                                 mcc_label = mcc_label,
                                 mcc_name_en = mcc_name_en,
                                 mcc_name_vi = mcc_name_vi,
                                 mcc_prob = mcc_prob,
                                 voice_en = voice_en,
                                 voice_vi = voice_vi,
                                 mlc_results = mlc_results)

        return render_template('index.html', readImg = '0', error="Invalid file type.")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000, threaded=True)
