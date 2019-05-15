import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from config import CONFIG
import json
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer

# load saved model architecture
json_file = open('model.conv1d.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

tokenizer = Tokenizer(num_words=CONFIG.getint('DEFAULT', 'VOCAB_SIZE'))

# open dictionary file created when training model
dictionary_file = open('dictionary.json', 'r', encoding='utf-8')
dictionary = json.load(dictionary_file)
dictionary_file.close()

# encode all words to utf-8
for key, value in dictionary.items():
    str(key).encode('utf-8')

# create tokenizer from dictionary
tokenizer.word_index = dictionary

loaded_model = tf.keras.models.model_from_json(loaded_model_json)

# load saved model weights
loaded_model.load_weights('model.conv1d.h5')

embeddings = loaded_model.layers[0]

print(embeddings.get_weights()[0][0])