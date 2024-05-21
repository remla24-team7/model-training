"""
Module to make the dataset by processing the data retrieved from Google Drive and saving it
"""

from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from lib_ml_remla import Preprocess

# check data locations with extraction from drive
with open("dataset/train.txt", "r", encoding="utf-8") as file:
    train = [line.strip() for line in file.readlines()[1:]]
raw_x_train = [line.split("\t")[1] for line in train]
raw_y_train = [line.split("\t")[0] for line in train]

with open("dataset/test.txt", "r", encoding="utf-8") as file:
    test = [line.strip() for line in file.readlines()]
raw_x_test = [line.split("\t")[1] for line in test]
raw_y_test = [line.split("\t")[0] for line in test]

with open("dataset/val.txt", "r", encoding="utf-8") as file:
    val = [line.strip() for line in file.readlines()]
raw_x_val = [line.split("\t")[1] for line in val]
raw_y_val = [line.split("\t")[0] for line in val]

tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
char_index = tokenizer.word_index
SEQUENCE_LENGTH = 200
x_train = Preprocess.preprocess(raw_x_train)# pad_sequences(tokenizer.texts_to_sequences(raw_x_train), maxlen=SEQUENCE_LENGTH)
x_val = Preprocess.preprocess(raw_x_val)# pad_sequences(tokenizer.texts_to_sequences(raw_x_val), maxlen=SEQUENCE_LENGTH)
x_test = Preprocess.preprocess(raw_x_test)# pad_sequences(tokenizer.texts_to_sequences(raw_x_test), maxlen=SEQUENCE_LENGTH)

encoder = LabelEncoder()

y_train = encoder.fit_transform(raw_y_train)
y_val = encoder.transform(raw_y_val)
y_test = encoder.transform(raw_y_test)

dump(x_train, 'outputs/x_train.joblib')
dump(x_val, 'outputs/x_val.joblib')
dump(x_test, 'outputs/x_test.joblib')
dump(y_test, 'outputs/y_test.joblib')
dump(y_train, 'outputs/y_train.joblib')
dump(y_val, 'outputs/y_val.joblib')
dump(char_index, 'outputs/char_index.joblib')
