from numpy import array
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import Flatten, LSTM
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt



dataset = pd.read_csv("iconsheet.csv")


def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


X = []
sentences = list(dataset["taskName"])
for sen in sentences: 
    X.append(preprocess_text(sen))

le = LabelEncoder()

lb = le.fit_transform(dataset['iconId'])

y = lb

# data = np.array(columnTransformer.fit_transform(dataset), dtype = np.str)

print(y)

# print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)



vocab_size = len(tokenizer.word_index) + 1

maxlen = 200



X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# print(X_test)

# print(vocab_size)
 
model = Sequential()
model.add(Embedding(vocab_size, 20, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


history = model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1, validation_split=0.2)
score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])




# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


# Save the model.
with open('keras/model.tflite', 'wb') as f:
  f.write(tflite_model)