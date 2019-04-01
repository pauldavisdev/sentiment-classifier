import json
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from config import CONFIG
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

tf.logging.set_verbosity(tf.logging.ERROR)

tokenizer = Tokenizer(num_words=CONFIG.getint('DEFAULT', 'VOCAB_SIZE'))

dictionary_file = open('dictionary.json', 'r')
dictionary = json.load(dictionary_file)
dictionary_file.close()

for key, value in dictionary.items():
    str(key).encode('utf-8')

tokenizer.word_index = dictionary

# load saved model architecture
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = tf.keras.models.model_from_json(loaded_model_json)

# load saved model weights
model.load_weights('model.h5')

while True:
    user_input = input('Input a sentence to be evaluated, or Enter to quit: ')

    if len(user_input) == 0:
        break

    words = text_to_word_sequence(user_input)

    cleaned_sentence = []
    
    stopwords_english = stopwords.words('english')

    stemmer = PorterStemmer()

    for word in words:
        word = stemmer.stem(word)
        if word in dictionary:
            if dictionary[word] < tokenizer.num_words and word not in stopwords_english: 
                cleaned_sentence.append(dictionary[word])
            else:
                word = '<STOPWORD>'
                cleaned_sentence.append(dictionary[word])
        else:
            word = '<UNK>'
            cleaned_sentence.append(dictionary[word])

    print(cleaned_sentence)

    cleaned_sentence = pad_sequences([cleaned_sentence], maxlen=140, padding='post')

    # predict sentiment of sentence
    pred = model.predict(cleaned_sentence)
    print(pred)

    sentiment = ['negative', 'positive']
    
    # and print it for the humons
    print("%s sentiment; %f%% confidence" % (sentiment[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))