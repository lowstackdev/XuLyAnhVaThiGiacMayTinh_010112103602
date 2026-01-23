from flask import Flask, render_template, request, session, redirect, url_for
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from ai.output.gradcamutils import generate_gradcam, load_and_preprocess_image, create_superimposed_image, save_image
from ai.output.bounding_box import draw_bounding_boxes
from common.pdf_report import generate_pdf_report
from common.bilingual_report_reader import get_bilingual_report_text
import pandas as pd

app = Flask(__name__,
            static_folder=os.path.join(os.path.dirname(__file__), 'static'),
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

app.secret_key = os.environ.get('SECRET_KEY', 'ocular_ai_secret_dev_key')
UPLOAD_FOLDER = Path(app.static_folder) / 'images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_PATH = os.path.join(BASE_DIR, 'ai', 'input', 'datasets', 'kaggle', 'rohitrawat25', 'combined-fundus-images', 'label_images.csv')
_label_df = pd.read_csv(CSV_PATH)
LABELS = sorted(_label_df['label'].unique())
MODEL_PATH = os.path.join(BASE_DIR, 'ai', 'models', 'checkpoints', 'disease_classify_model.h5')
logger.info(f"Loading retinal disease model from {MODEL_PATH}")
MODEL = load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_uploads():
    if UPLOAD_FOLDER.exists():
        for file in UPLOAD_FOLDER.iterdir():
            if file.is_file():
                try:
                    file.unlink()
                except Exception as e:
                    logger.error(f"Cleanup error for {file}: {e}")

def process_image_analysis(file_path):
    try:
        file_path = Path(file_path)

        # 1. Image Preprocessing & Prediction
        img = load_img(str(file_path), target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = MODEL.predict(x)
        probs_array = preds[0]
        top_indices = probs_array.argsort()[::-1]
        probs = [f"{probs_array[i]:.2f}" for i in top_indices]
        labels = [LABELS[i] for i in top_indices]
        main_label = LABELS[np.argmax(probs_array)]

        # 2. Visual AI Features (Grad-CAM)
        gradcam_filename = f"{file_path.stem}_gradcam.jpg"
        gradcam_path = UPLOAD_FOLDER / gradcam_filename
        target_idx = np.argmax(preds[0])
        generate_gradcam(MODEL, str(file_path), target_idx, str(gradcam_path))

        # 3. Structural Analysis (Bounding Boxes)
        highlight_name = f"{file_path.stem}_highlight.png"
        draw_bounding_boxes(
            str(gradcam_path),
            str(UPLOAD_FOLDER),
            main_label,
            labels[1:],
            float(probs[0]),
            None
        )

        actual_highlight = f"{file_path.stem}_gradcam_highlighted.png"

        return {
            'success': True,
            'label': main_label,
            'probs': probs,
            'labs': labels,
            'gradcam_url': gradcam_filename,
            'highlighted_url': actual_highlight
        }
    except Exception as e:
        logger.error(f"AI Analysis failed: {e}")
        return {'success': False, 'error': str(e)}

@app.context_processor
def inject_globals():
    return {'now': datetime.now()}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        if not session.get('data'):
            cleanup_old_uploads()

    if request.method == 'POST':
        # 1. Validation
        file = request.files.get('filename')
        if not file or file.filename == '':
            session['error'] = 'No image selected for analysis.'
            return redirect(url_for('index'))

        if not allowed_file(file.filename):
            session['error'] = 'Unsupported file format. Use PNG, JPG, or JPEG.'
            return redirect(url_for('index'))

        # 2. Save & Process
        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        filename = secure_filename(file.filename)
        file_path = UPLOAD_FOLDER / filename
        file.save(str(file_path))

        # Execute AI Pipeline
        analysis = process_image_analysis(file_path)

        if not analysis['success']:
            session['error'] = f"Analysis Error: {analysis.get('error')}"
            return redirect(url_for('index'))

        # 3. Report Generation & Session Management
        report_name = f"{file_path.stem}_report.pdf"
        report_path = UPLOAD_FOLDER / report_name

        speech_en = f"Diagnostic report for retinal analysis. The detected condition is {analysis['label']} with a confidence of {analysis['probs'][0]} percent."
        speech_vi = f"Báo cáo chẩn đoán võng mạc. Tình trạng được phát hiện là {analysis['label']} với độ tin cậy là {analysis['probs'][0]} phần trăm."

        report_data = {
            'date': datetime.now().strftime('%d %b %Y, %H:%M'),
            'disease_label': analysis['label'],
            'max_prob': analysis['probs'][0],
            'prob': analysis['probs'],
            'labs': analysis['labs'],
            'report_filename': report_name,
            'highlighted_image_url': url_for('static', filename=f'images/{analysis["highlighted_url"]}'),
            'speech_text_en': speech_en,
            'speech_text_vi': speech_vi
        }

        try:
            generate_pdf_report(str(report_path), data=report_data)

            bilingual_data = get_bilingual_report_text(str(report_path), target_lang="vi")
            if bilingual_data and str(bilingual_data.get('en', '')).strip():
                report_data['speech_text_en'] = str(bilingual_data['en'])[:500]
                report_data['speech_text_vi'] = str(bilingual_data['translated'])[:500]
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")

        for key in ['speech_text_en', 'speech_text_vi']:
            if not report_data.get(key) or not str(report_data[key]).strip():
                report_data[key] = speech_en if 'en' in key else speech_vi
            report_data[key] = str(report_data[key])[:500]

        logger.info(f"Final Data Ready. EN Length: {len(report_data['speech_text_en'])}")

        # 4. Store Results Securely
        session.clear()
        session['data'] = report_data
        session['readImg'] = '1'
        session.modified = True
        return redirect(url_for('index'))

    error = session.pop('error', None)
    data = session.pop('data', None)
    read_img = session.pop('readImg', '0')

    return render_template('index.html', data=data, error=error, readImg=read_img)

if __name__ == '__main__':
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    app.run(debug=True, use_reloader=False, port=8000, threaded=True)
