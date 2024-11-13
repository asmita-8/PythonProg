###(i) Generate a dataset of 1000 English sentences describing fictitious movie reviews (need not be exact movie reviews).
###Attach a positive or negative label for each sample (review) reflecting positive or negative sentiment about the movie based on the review description.
###(ii)  Use the generated dataset and train a RNN classifier to predict the movie sentiment based on movie review descriptions.

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

##some example phrases
positive_phrases = [
    "an incredible journey", "beautifully shot", "masterfully directed", "excellent character development",
    "an unforgettable experience", "a heartwarming story", "brilliantly acted", "truly inspiring",
    "a visual masterpiece", "outstanding performances", "a delight from start to finish",
    "left me speechless", "a must-watch", "captures the essence of storytelling", "amazingly crafted scenes"
]
negative_phrases = [
    "disappointingly dull", "lacks depth", "a predictable plot", "poorly written dialogue",
    "fails to impress", "a missed opportunity", "slow and boring", "uninteresting characters",
    "a waste of time", "below expectations", "falls short in every aspect", "overly dramatic and tedious",
    "hard to sit through", "did not live up to the hype", "forgettable performances"
]
num_samples = 1000
data = []
for _ in range(num_samples):
    if np.random.random() > 0.5:
        label = "Positive"
        phrase = np.random.choice(positive_phrases)
    else:
        label = "Negative"
        phrase = np.random.choice(negative_phrases)
    review = f"{phrase};Recommended!" if label=="Positive" else f"{phrase}; Not Recommended."
    data.append([review, label])
df = pd.DataFrame(data, columns=["Review", "Label"])
df.to_csv('movie_reviews_dataset.csv', index=False)

###loading the dataset
df = pd.read_csv('movie_reviews_dataset.csv')

###splitting into training and test sets
train_reviews, test_reviews, train_labels, test_labels = train_test_split(
    df['Review'].values, df['Label'].values, test_size=0.2, random_state=42
)

###converting labels to binary (1 for positive, 0 for negative)
train_labels = np.array([1 if label == "Positive" else 0 for label in train_labels])
test_labels = np.array([1 if label == "Positive" else 0 for label in test_labels])

###tokenize the text data
vocab_size = 10000
max_length = 100    #maximum length of reviews

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_reviews)
train_sequences = tokenizer.texts_to_sequences(train_reviews)
test_sequences = tokenizer.texts_to_sequences(test_reviews)

###Pading the sequences to ensure uniform input length
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

###Define the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64, input_length=max_length),
    tf.keras.layers.SimpleRNN(128, return_sequences=False),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  #sigmoid for binary classification
])

###compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

###train the model
model.fit(train_padded, train_labels, epochs=10, validation_data=(test_padded, test_labels))

###evaluate on the test set
loss, accuracy = model.evaluate(test_padded, test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

##Make a prediction on a new review
new_review = ["The plot was beautifully executed with stunning visuals."]
new_seq = tokenizer.texts_to_sequences(new_review)
new_padded = pad_sequences(new_seq, maxlen=max_length, padding='post', truncating='post')
prediction = model.predict(new_padded)

##Convert prediction to positive/negative
predicted_label = "positive" if prediction[0][0] > 0.5 else "negative"
print(f"Predicted sentiment: {predicted_label}")


###(i) Generate an English text data of a long paragraph.
###(ii) Using the generated English text data, implement a character level language model, i.e. predicting the next character in a sequence based on past characters,
###using a LSTM network.

import string
import string
import numpy as np
from tensorflow.keras.utils import to_categorical, pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

raw_text = ("In a world where technology continues to evolve at an unprecedented pace, "
            "human life is undergoing significant transformation across nearly every facet "
            "of existence. Communication has become instantaneous, connecting people across "
            "continents with a simple tap or click, creating a global community more interconnected "
            "than ever before. Industries, from healthcare to transportation, are harnessing the "
            "power of artificial intelligence, machine learning, and big data to streamline "
            "operations, enhance productivity, and offer services that were once thought to belong "
            "in the realm of science fiction.")

# Function to clean the text data
def clean_data(text):
    tokens = text.split()
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t.lower() for t in tokens]
    tokens = ' '.join(tokens)
    return tokens
raw_text = clean_data(raw_text)
print(raw_text)

# Define sequence length and create sequences list
length = 10
sequences = []

# Function to create character sequences
def create_seq(raw_text):
    for i in range(length, len(raw_text)):
        sequences.append(raw_text[i - length:i + 1])
    print('Total Sequences', len(sequences))
create_seq(raw_text)
print(sequences)

# Create a character mapping dictionary
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
print(mapping)

# Encoding each character sequence according to the mapping
encoded_sequences = []
for line in sequences:  # Use sequences here instead of lines
    encode_seq = [mapping[char] for char in line]
    encoded_sequences.append(encode_seq)
print(encoded_sequences[0])

#Define vocabulary size, encode sequences, etc.
vocab_size = len(mapping)
print(vocab_size)  # Expected output: 22

# Prepare input (X) and output (y) arrays
encoded_sequences = np.array(encoded_sequences)
X, y = encoded_sequences[:, :-1], encoded_sequences[:, -1]
print(X[0])  # Example output: [9 12  0  1  0 20 13 15 10  4]
print(y[0])  # Example output: 0
print(X.shape)  # (455, 10)
print(y.shape)  # (455,)

# One-hot encode the input sequences
onehot_encoded_seq = [to_categorical(x, num_classes=vocab_size) for x in X]
X = np.array(onehot_encoded_seq)
y = to_categorical(y, num_classes=vocab_size)
print(X.shape)  # (455, 10, 22)
print(y.shape)  # (455, 22)

# Define the LSTM model
def define_model(X):
    model = Sequential()  # Initialize a Sequential model
    model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))  # Add LSTM layer
    model.add(Dense(50, activation='relu'))  # Add a dense layer
    model.add(Dense(100, activation='relu'))  # Add another dense layer
    model.add(Dense(150, activation='relu'))  # Add another dense layer
    model.add(Dense(vocab_size, activation='softmax'))  # Add output layer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Compile the model
    model.summary()  # Print the model summary
    return model
model = define_model(X)
model.fit(X, y, epochs=200, verbose=2)

#Generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_txt = clean_data(seed_text)
    for _ in range(n_chars):
        # Encode the text as integers
        encoded_seq = [mapping[char] for char in in_txt]
        # Truncate sequences to a fixed length
        encoded_seq = pad_sequences([encoded_seq], maxlen=seq_length, truncating='pre')
        # One-hot encode the sequence
        encoded_seq = to_categorical(encoded_seq, num_classes=len(mapping))
        # Reshape for prediction
        encoded_seq = encoded_seq.reshape(1, encoded_seq.shape[1], encoded_seq.shape[2])
        # Predict the next character
        yhat = model.predict(encoded_seq, verbose=0)
        yhat = np.argmax(yhat, axis=-1)[0]
        # Map integer back to character
        out_char = ''
        for char, i in mapping.items():
            if i == yhat:
                out_char = char
                break
        # Append the predicted character to the input text
        in_txt += out_char
    return in_txt
# Example usage
seed_text = "In a world"
generated_text = generate_seq(model, mapping, seq_length=10, seed_text=seed_text, n_chars=20)
print(generated_text)

