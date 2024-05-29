"""
Module for training the neural network model using Keras.
"""

from joblib import load
from keras.src.models import Sequential
from keras.src.layers import Embedding, Conv1D, MaxPooling1D, Dropout, Flatten, Dense

def load_data():
    x_train = load('outputs/x_train.joblib')
    y_train = load('outputs/y_train.joblib')
    x_val = load('outputs/x_val.joblib')
    y_val = load('outputs/y_val.joblib')
    char_index = load('outputs/char_index.joblib')
    return x_train, y_train, x_val, y_val, char_index


def build_model(voc_size, categories, embedding_dimension):
    model = Sequential()
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

    model.add(Dense(len(categories) - 1, activation='sigmoid'))
    return model


def train_model(x_train, y_train, x_val, y_val, params, char_index):
    voc_size = len(char_index.keys())
    model = build_model(voc_size, params['categories'], params['embedding_dimension'])
    model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])
    hist = model.fit(x_train, y_train, batch_size=params['batch_train'], epochs=params['epoch'], shuffle=True,
                     validation_data=(x_val, y_val))
    model.save('outputs/model.h5')
    return model, hist


if __name__ == "__main__":
    x_train, y_train, x_val, y_val, char_index = load_data_test()
    params = {'loss_function': 'binary_crossentropy',
              'optimizer': 'adam',
              'sequence_length': 200,
              'batch_train': 5000,
              'batch_test': 5000,
              'categories': ['phishing', 'legitimate'],
              'char_index': None,
              'epoch': 30,
              'embedding_dimension': 50,
              'dataset_dir': "dataset/small_dataset/"
              }
    model, hist = train_model(x_train, y_train, x_val, y_val, params, char_index)
