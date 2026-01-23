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

app = Flask(__name__,
            static_folder=os.path.join(os.path.dirname(__file__), 'static'),
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.context_processor
def inject_globals():
    return {'now': datetime.now()}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', readImg = '0')
    if request.method == 'POST':
        return render_template('index.html', readImg = '1')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000, threaded=True)
