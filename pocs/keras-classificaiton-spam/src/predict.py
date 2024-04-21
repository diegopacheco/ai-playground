import sys
import os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, '.')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # less verbose logs

from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
import pickle
from src.model import *

# Load the Tokenizer
tokenizer = None
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# load model from file
model = keras.models.load_model('spam_model.keras')

def predict(text):
    new_sequence = tokenizer.texts_to_sequences([text])[0]
    padded_new = pad_sequences([new_sequence], maxlen=max_length)

    prediction = model.predict(padded_new,verbose=0)
    if prediction[0][1] > 0.5:
        return "Spam"
    else:
        return "Not Spam"

print("Win a trip to Hawaii this summer! Predition: " + 
      str(predict("Win a trip to Hawaii this summer!")))

print("John I cannot go to the party. Predition:" +
      str(predict("John I cannot go to the party.")))

print("Honey I'm running late, please start cooking the dinner. " +
      str(predict("Honey I'm running late, please start cooking the dinner.")))