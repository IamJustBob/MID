import pandas as pd
import keras
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle

num_epochs = 150
batch_size = 128

data = pd.read_csv('dataset.csv')

train, test = train_test_split(data, test_size=0.2)
train_text, train_labels = train['text'], train['label']
test_text, test_labels = test['text'], test['label']

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_text)

train_text = tokenizer.texts_to_sequences(train_text)
test_text = tokenizer.texts_to_sequences(test_text)

train_text = keras.preprocessing.sequence.pad_sequences(train_text)
test_text = keras.preprocessing.sequence.pad_sequences(test_text)

encoder = LabelEncoder()
encoder.fit(train_labels)
train_labels = encoder.transform(train_labels)
test_labels = encoder.transform(test_labels)

train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

with open('tokenizer.pickle', 'wb') as handle:
	pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('encoder.pickle', 'wb') as handle:
	pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(len(tokenizer.word_index)+1)

model = Sequential([
	Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=512),
	Bidirectional(LSTM(256, return_sequences=True, dropout=0.5)),
	LSTM(256, dropout=0.5),
	Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
	Dropout(0.5),
	Dense(len(train_labels[0]), activation='softmax')
])

def clr(epoch):
  base_lr = 0.0001
  max_lr = 0.01
  step_size = 10
  cycle = np.floor(1 + epoch / (2 * step_size))
  x = np.abs(epoch / step_size - 2 * cycle + 1)
  lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
  return lr

early_stopping = EarlyStopping(monitor='val_loss', patience=11)
model_checkpoint = ModelCheckpoint('best_model-l.h5', monitor='val_accuracy', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
clr_callback = keras.callbacks.LearningRateScheduler(clr)

callbacks = [early_stopping, model_checkpoint, reduce_lr, clr_callback]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(train_text, train_labels, epochs=num_epochs, validation_data=(test_text, test_labels), callbacks=callbacks, batch_size=batch_size)

model.save("MID-l.h5")

plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('model_accuracy.png')

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('model_loss.png')


