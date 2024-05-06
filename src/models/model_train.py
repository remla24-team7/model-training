"""
Module for training the neural network model using Keras.
"""

from joblib import load
from keras.src.models import Sequential
from keras.src.layers import Embedding, Conv1D, MaxPooling1D, Dropout, Flatten, Dense

x_train = load('outputs/x_train.joblib')
y_train = load('outputs/y_train.joblib')
x_val = load('outputs/x_val.joblib')
y_val = load('outputs/y_val.joblib')
char_index = load('outputs/char_index.joblib')

params = {'loss_function': 'binary_crossentropy',
          'optimizer': 'adam',
          'sequence_length': 200,
          'batch_train': 5000,
          'batch_test': 5000,
          'categories': ['phishing', 'legitimate'],
          'char_index': None,
          'epoch': 30,
          'embedding_dimension': 50,
          'dataset_dir': "../dataset/small_dataset/"}

model = Sequential()
voc_size = len(char_index.keys())
print(f"voc_size: {voc_size}")
model.add(Embedding(voc_size + 1, 50))

model.add(Conv1D(128, 3, activation='tanh'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.2))

model.add(Conv1D(128, 7, activation='tanh', padding='same'))
model.add(Dropout(0.2))

model.add(Conv1D(128, 5, activation='tanh', padding='same'))
model.add(Dropout(0.2))

model.add(Conv1D(128, 3, activation='tanh', padding='same'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.2))

model.add(Conv1D(128, 5, activation='tanh', padding='same'))
model.add(Dropout(0.2))

model.add(Conv1D(128, 3, activation='tanh', padding='same'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.2))

model.add(Conv1D(128, 3, activation='tanh', padding='same'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(len(params['categories']) - 1, activation='sigmoid'))

model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])

hist = model.fit(x_train, y_train,
                 batch_size=params['batch_train'],
                 epochs=params['epoch'],
                 shuffle=True,
                 validation_data=(x_val, y_val)
                 )

model.summary()

model.save('outputs/model.h5')
