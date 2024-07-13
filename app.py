from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
from preprocess import preprocess_image
from model import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load the pre-trained model
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        image = preprocess_image(filepath)
        prediction = model.predict(np.array([image]))
        result = np.argmax(prediction)
        return f'The plant is {"healthy" if result == 0 else "diseased"}'

if __name__ == '__main__':
    app.run(debug=True)
