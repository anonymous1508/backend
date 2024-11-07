# backend/app.py
from flask import Flask, request, jsonify
import base64
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load a dummy model (assuming 'e_waste_model.h5' exists)
model = tf.keras.models.load_model('e_waste_model.h5')

def preprocess_image(image_data):
    image = Image.open(image_data).resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    image_data = base64.b64decode(data['image'].split(',')[1])
    image = preprocess_image(image_data)
    
    predictions = model.predict(image)
    e_waste_type = np.argmax(predictions)

    types = {0: 'Battery', 1: 'Circuit Board', 2: 'Plastic', 3: 'Screen'}
    result = types.get(e_waste_type, 'Unknown')
    return jsonify({'type': result})

if __name__ == '__main__':
    app.run(port=5000)
