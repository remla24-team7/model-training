import urllib.request
import zipfile
from tensorflow.keras.preprocessing.text import Tokenizer

URL = 'gdrive://48900e618890f4ce97201d1592d45045/dataset' #todo: fix repro get data from remote

EXTRACT_DIR = "../data"

x, y, z = urllib.request.urlretrieve(URL)
# with zipfile.ZipFile(zip_path, "r") as f:
#     f.extractall(EXTRACT_DIR)

def train_test_split():
    train = [line.strip() for line in open("../data/data/DL Dataset/train.txt", "r").readlines()[1:]]
    raw_x_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]

    test = [line.strip() for line in open("../data/data/DL Dataset/test.txt", "r").readlines()]
    raw_x_test = [line.split("\t")[1] for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]

    val=[line.strip() for line in open("../data/data/DL Dataset/val.txt", "r").readlines()]
    raw_x_val=[line.split("\t")[1] for line in val]
    raw_y_val=[line.split("\t")[0] for line in val]

    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
    char_index = tokenizer.word_index
    sequence_length=200
    x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train), maxlen=sequence_length)
    x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val), maxlen=sequence_length)
    x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test), maxlen=sequence_length)

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    return x_train, x_val, x_test, y_train, y_val, y_test, char_index