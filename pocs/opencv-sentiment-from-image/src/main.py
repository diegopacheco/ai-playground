import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_model.h5')  # replace with the path to your model

# Define the list of emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the image
img = cv2.imread('image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    # Extract the region of interest
    roi_gray = gray[y:y+h, x:x+w]

    # Normalize the face
    roi_gray = roi_gray / 255.0

    # Resize the face to 48x48 pixels
    roi_gray = cv2.resize(roi_gray, (48, 48))

    # Reshape the face to match the input shape of the model
    roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

    # Predict the emotion
    prediction = emotion_model.predict(roi_gray)
    emotion = emotion_labels[np.argmax(prediction)]

    # Display the emotion on the image
    cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

# Display the image
cv2.imshow('Emotion Detection', img)
cv2.waitKey(5000)
cv2.destroyAllWindows()