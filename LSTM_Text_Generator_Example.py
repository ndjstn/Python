'''
    This file will allow for text to be generated. This effectively takes a url of text, 
    but could be replaced with just a text document. This is done by modifying the file_path
    variable. This may be useful to utilize in conjuction with argparse, and possibly with 
    a sum of text from pandas column. Adding stop words feature or finding a way to penalize
    stop words will make this come across more realistic. This in nice though as it is language
    (Human) agnostic. I apologize in advance if there are people opposed to me using the Bible
    for this project. It is a large dataset and freely available from Guttenberg.
    
    Inputs: URL/Text
    Outputs: Predicted text body of users 
    
    Suggested 
    TODO: Create install schema
    TODO: Implement argparse
    TODO: Speed up code using generators, parallel processing, and proper GPU settings.
'''

# Required imports
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM
  
# Attempt to read over the file
try:
    file_path = tf.keras.utils.get_file('bible.txt',
            'https://www.gutenberg.org/files/10/10-0.txt')
    text = open(file_path, 'rb').read().decode(encoding='utf-8').lower()

except Exception as e:
    
    print(f"There was an error decoding that file. -- {e}.")
    
# Sort out the unique text for the model.
characters = sorted(set(text)) 

# Make Translators to convert from str to int/vice-versa
character_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_character = dict((i, c) for i, c in enumerate(characters))

# Length of input text to look back at for prediction of next.
Length_Of_Characters_To_Sequence = 60

# Space between sentences that are being analyzed odd or prime numbers preferably (entropy).
Space_Between_Squences = 7

# Lists to hold values
sentences = []
next_char = []

# Iter over the text 
for i in range(0, len(text) - Length_Of_Characters_To_Sequence, Space_Between_Squences):
    sentences.append(text[i: i + Length_Of_Characters_To_Sequence])
    next_char.append(text[i + Length_Of_Characters_To_Sequence])

# Arrays to hold place values
x = np.zeros((len(sentences), Length_Of_Characters_To_Sequence,
              len(characters)), dtype = bool)
y = np.zeros((len(sentences),
              len(characters)), dtype = bool)

# Fill arrays
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, character_to_index[char]] = 1
    y[i, character_to_index[next_char[i]]] = 1

# Create a Sequential model object
model = Sequential()

# Add LSTM values to the model
model.add(LSTM(256,
        input_shape=(Length_Of_Characters_To_Sequence,
        len(characters))))

model.add(Dense(len(characters)))

model.add(Activation('softmax'))

# Set learning rate and factor for loss
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate = 0.01))

# Fit the model
model.fit(x, y, 
          batch_size = 256, 
          epochs = 10)

# Functions
def sample_set(predictions, Distance_From_Normal=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / Distance_From_Normal
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)

def generate_text(length, Distance_From_Normal):
    start_index = random.randint(0, len(text) - Length_Of_Characters_To_Sequence - 1)
    generated = ''
    sentence = text[start_index: start_index + Length_Of_Characters_To_Sequence]
    generated += sentence
    for i in range(length):
        x_predictions = np.zeros((1, Length_Of_Characters_To_Sequence, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, character_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose = 0)[0]
        next_index = sample_set(predictions,
                                 Distance_From_Normal)
        next_character = index_to_character[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

# Here is how the model can be used.
print(generate_text(300, 0.2), "\n")
