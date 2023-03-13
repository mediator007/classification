import pandas as pd
import numpy as np
import re

from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
import gensim

# Load data from xlsx file
data = pd.read_excel('data.xlsx', usecols=['Question', 'Answer'])
text = data['A'].tolist()
labels = data['B'].tolist()

# Clean text data
def text_cleaner(text):
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

text = [text_cleaner(t) for t in text]

# # Tokenize text data
tokenizer = Tokenizer(num_words=5000, split=' ')
tokenizer.fit_on_texts(text)
X = tokenizer.texts_to_sequences(text)
X = pad_sequences(X)

# Convert labels to numerical values
unique_labels = list(set(labels))
label_dict = {label: i for i, label in enumerate(unique_labels)}
y = [label_dict[label] for label in labels]
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Keras model
model = Sequential()
model.add(Embedding(5000, 128, input_length=X.shape[1]))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(unique_labels), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=64)

model.save('model.h5')
model = load_model('model.h5')

test_data = ['лк'] # Question for answer
test_sequences = tokenizer.texts_to_sequences(test_data)
test_sequences_padded = pad_sequences(test_sequences)
predictions = model.predict(test_sequences_padded)

print(predictions)
class_labels = np.argmax(predictions, axis=1)

# Print the predicted class labels for the test data
print(class_labels)