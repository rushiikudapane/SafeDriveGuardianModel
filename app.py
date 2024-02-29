from flask import Flask, request
import cv2
import numpy as np
from keras.models import load_model

model = load_model('models/cnnCat2.h5')

app = Flask(__name__)

model = load_model('models/cnncat2.h5')

score = 0
THRESHOLD_SCORE = 20

face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
rpred = [99]
lpred = [99]

def detect_drowsiness(frame):
    global score
    global rpred, lpred

    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = np.argmax(model.predict(r_eye), axis=1)

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = np.argmax(model.predict(l_eye), axis=1)

    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Ensure score doesn't go below 0
    score = max(0, score)

    return frame



@app.route('/predict', methods=['POST'])
def predict():
    
    image_file = request.files.get('image')

    if not image_file:
        return 'No image found', 400
    
    frame = detect_drowsiness(image_file)

    # Check if the score exceeds the threshold
    if score > THRESHOLD_SCORE:
        # Trigger an alert (you can add your own alert mechanism here)
        print("Drowsiness detected! Triggering alert.")

    return {'score': score}



if __name__ == '__main__':
    app.run(debug=True, port=3000)
