import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split

# Load names and labels from the CSV file
df = pd.read_csv('names.csv')  # Update with your actual file path
names = df['Full.name'].tolist()
labels = df['Sex'].tolist()

# Create a character vocabulary
characters = set(char for name in names for char in name)
char_vocab = {char: idx + 1 for idx, char in enumerate(characters)}  # Add 1 to reserve 0 for padding

# Convert names to sequences of character indices
max_length = max(len(name) for name in names)
name_sequences = [[char_vocab[char] for char in name] for name in names]

# Pad sequences to have the same length
padded_sequences = np.array([np.pad(seq, (0, max_length - len(seq)), 'constant') for seq in name_sequences])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=len(char_vocab) + 1, output_dim=10, input_length=max_length))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

# Evaluate the model on new names
new_names = ['Джумаев Эсен', 'Мирлан уулу Жоомарт', 'Маликова Жаннат']
new_name_sequences = [[char_vocab.get(char, 0) for char in name] for name in new_names]  # Convert to sequences
padded_new_sequences = np.array([np.pad(seq, (0, max_length - len(seq)), 'constant') for seq in new_name_sequences])
predictions = model.predict(padded_new_sequences)

# Print predictions
for name, prediction in zip(new_names, predictions):
    gender = 'Male' if prediction >= 0.5 else 'Female'
    print(f"{name}: {gender} ({prediction[0]:.4f})")
