from csv_handler import read_csv, write_csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, one_hot
from keras.utils import to_categorical
from neural_network import NeuralNetwork
from preprocessor import Preprocessor
from keras import optimizers
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import time, os, json
import pandas as pd
import pickle
from config import CONFIG
from nltk.corpus import stopwords

def prepare_data():

    num_csv_rows = CONFIG.getint('DEFAULT', 'TRAINING_SIZE')
    
    max_len = 140

    train = read_csv(num_csv_rows)

    trainX, trainY = train['tweet'].values, train['polarity'].values

    trainY = np.where(trainY == 4, 1, trainY)

    preprocessor = Preprocessor()

    trainX = preprocessor.clean_texts(trainX)

    vocab_size = CONFIG.getint('DEFAULT', 'VOCAB_SIZE')

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNUSED>")

    tokenizer.fit_on_texts(trainX)

    # 0 reserved for padding, 1 reserved for unknown words
    # 2 reserved for unused words (least frequent), 3 reserved for stopwords
    tokenizer.word_index = { k: (v + 1) for k, v in tokenizer.word_index.items() } 
    tokenizer.word_index["<UNK>"] = 1
    tokenizer.word_index["<UNUSED>"] = 2

    trainX = tokenizer.texts_to_sequences(trainX)

    trainX = pad_sequences(trainX, maxlen=max_len, padding='post')

    dictionary = tokenizer.word_index

    with open('dictionary.json', 'w', encoding='utf-8') as dictionary_file:
        json.dump(dictionary, dictionary_file, ensure_ascii=False)
    
    return trainX, trainY, vocab_size, max_len

def plot_graph(history):

    history_dict = history.history

    history_dict.keys()

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.grid(b=True, visible=True)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b', label='Training Loss', linestyle='--')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    #plt.show()
    plt.subplot(1, 2, 2)
    #plt.clf()   # clear figure
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    plt.plot(epochs, acc, 'b', label='Training Accuracy', linestyle='--')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def create_csv_dataframe(history, results):
    history_dict = history.history

    df = pd.DataFrame.from_dict(history_dict)
    df['test_loss'] = results[0]
    df['test_acc'] = results[1]
    df['input nodes'] = CONFIG.getint('DEFAULT', 'EMBEDDING_OUTPUT')
    df['hidden nodes'] = CONFIG.getint('DEFAULT', 'HIDDEN')
    df['output nodes'] = CONFIG.getint('DEFAULT', 'OUTPUT')
    df['epochs'] = CONFIG.getint('DEFAULT', 'EPOCHS')
    df['batch size'] = CONFIG.getint('DEFAULT', 'BATCH_SIZE')
    df['patience'] = CONFIG.getint('DEFAULT', 'PATIENCE')
    df = df.round(decimals=4)

    return df


def main():
    
    trainX, trainY, vocab_size, max_len = prepare_data()

    # split training data up into training and test sets
    trainX, testX, trainY, testY = train_test_split(trainX, trainY, test_size=0.1)

    # file = open('data_01_numwords30000.pkl', 'rb')
    # trainX = pickle.load(file)
    # trainY = pickle.load(file)
    # testX = pickle.load(file)
    # testY = pickle.load(file)
    # vocab_size = pickle.load(file)
    # max_len = 140
    # file.close()

    # file = open('data_01_numwords30000.pkl','wb')
    # pickle.dump(trainX, file)
    # pickle.dump(trainY, file)
    # pickle.dump(testX, file)
    # pickle.dump(testY, file)
    # pickle.dump(vocab_size, file)
    # pickle.dump(max_len, file)
    # file.close()

    # exit()

    trainY = to_categorical(trainY)

    testY = to_categorical(testY)

    neural_network = NeuralNetwork()

    neural_network.create_model(vocab_size, max_len)

    neural_network.model.summary()
    
    neural_network.model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
   
    neural_network.model.save('model.h5')

    val_size = CONFIG.getint('DEFAULT', 'VALIDATION_SIZE')

    x_val = trainX[:val_size]
    partial_x_train = trainX[val_size:]

    y_val = trainY[:val_size]
    partial_y_train = trainY[val_size:]

    history = neural_network.fit_model(partial_x_train, partial_y_train, 
                                        x_val, y_val)

    results = neural_network.model.evaluate(testX, testY)

    print(neural_network.model.metrics_names)

    print(results)

    model_json = neural_network.model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)

    neural_network.model.save_weights('model.h5')

    #plot_graph(history)

    write_csv(create_csv_dataframe(history, results))

if __name__ == '__main__':
    for i in range(CONFIG.getint('DEFAULT', 'RUNS_PER_EPOCH')):
        main()