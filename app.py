from flask import Flask, request
import cv2
import numpy as np
from keras.models import load_model

model = load_model('models/cnnCat2.h5')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files.get('image')

    if not image_file:
        return 'No image found', 400
    
    # pre process image, modify this if we have used any other preprocessing technique
    # write preporocessing logic here
    image = np.fromstring(image_file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (24, 24))
    image = image / 255
    image = image.reshape(24, 24, -1)
    image = np.expand_dims(image, axis=0)


    prediction = model.predict(image)
    # label = open if score > 15 else false
    label = 'Open' if np.argmax(prediction) == 1 else 'Closed'
    return label



if __name__ == '__main__':
    app.run(debug=True, port=3000)
