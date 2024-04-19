import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD


# Define the path to your images and the emotions
#
# Dataset: https://www.kaggle.com/datasets/msambare/fer2013
#
base_dir = 'data/train'
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load and preprocess the images
images = []
labels = []
for i, emotion in enumerate(emotions):
    emotion_dir = os.path.join(base_dir, emotion)
    for filename in os.listdir(emotion_dir):
        if filename.endswith('.jpg'):  # adjust if your images have a different format
            img = cv2.imread(os.path.join(emotion_dir, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            img = cv2.resize(img, (48, 48))  # resize to 48x48
            images.append(img)
            labels.append(i)

# Convert to numpy arrays and normalize the images
images = np.array(images) / 255.0
labels = np.array(labels)

# Reshape the images to fit the model input shape
images = images.reshape((-1, 48, 48, 1))

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# Fit the datagen on your data
datagen.fit(images)

# Define your model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(emotions), activation='softmax')
])

# Compile your model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create the SGD optimizer
#learning_rate = 0.01
#momentum = 0.9
#sgd = SGD(learning_rate=learning_rate, momentum=momentum)
#model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train your model
model.fit(datagen.flow(images, labels), epochs=15)

# Save the model
model.save('emotion_model.h5')
print('Model saved to emotion_model.h5')