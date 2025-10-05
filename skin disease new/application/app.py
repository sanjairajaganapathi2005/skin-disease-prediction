from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import os
import io
from PIL import Image

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

# Define class names
class_names = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis Photos', 'Bullous Disease Photos', 'Cellulitis Impetigo and other Bacterial Infections', 'Eczema Photos', 'Exanthems and Drug Eruptions', 'Hair Loss Photos Alopecia and other Hair Diseases', 'Herpes HPV and other STDs Photos', 'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases', 'Melanoma Skin Cancer Nevi and Moles', 'Nail Fungus and other Nail Disease', 'Poison Ivy Photos and other Contact Dermatitis', 'Psoriasis pictures Lichen Planus and related diseases', 'Scabies Lyme Disease and other Infestations and Bites', 'Seborrheic Keratoses and other Benign Tumors', 'Systemic Disease', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives', 'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum and other Viral Infections']

# Try to find a model file in the model directory if specific file is not present
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
PREFERRED_MODEL = os.path.join(MODEL_DIR, 'Resnet50.keras')

def find_model_path():
    # Return preferred if present, else pick first .h5 or .keras file in model dir
    if os.path.exists(PREFERRED_MODEL):
        return PREFERRED_MODEL
    if not os.path.isdir(MODEL_DIR):
        return None
    for fname in os.listdir(MODEL_DIR):
        if fname.lower().endswith(('.h5', '.keras')):
            return os.path.join(MODEL_DIR, fname)
    return None

MODEL_PATH = find_model_path()
model = None

if MODEL_PATH is None:
    raise RuntimeError('No model file found in model/ directory. Place an .h5 or .keras file there.')

# Load model
model = load_model(MODEL_PATH)


@app.route('/')
def index():
    return render_template('index.html')


def preprocess_image_file(file_bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    contents = file.read()
    try:
        img_array = preprocess_image_file(contents, target_size=(224, 224))
    except Exception as e:
        return jsonify({'error': f'Invalid image file: {e}'}), 400

    prediction = model.predict(img_array)
    print("Raw model prediction:", prediction)
    class_index = int(np.argmax(prediction))
    class_name = class_names[class_index] if class_index < len(class_names) else 'Unknown'
    probability = float(prediction[0][class_index]) if class_index > 0 else 0.0
    print("class_index", class_index, "class_name", class_name, "probability", probability)
    return jsonify({
        'predicted_class': class_index,
        'predicted_class_name': class_name,
    })


if __name__ == '__main__':
    # Use Flask's built-in server for simple runs
    app.run(host='0.0.0.0', port=8000, debug=True)
