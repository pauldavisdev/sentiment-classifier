from csv_handler import read_csv, write_csv
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, one_hot
from neural_network import NeuralNetwork
from preprocessor import Preprocessor
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import time, os
import pandas as pd
import pickle
from config import CONFIG

def prepare_data():

    num_csv_rows = CONFIG.getint('DEFAULT', 'TRAINING_SIZE')
    
    max_len = 140

    train, test = read_csv(num_csv_rows)

    testX, testY = test['tweet'].values, test['polarity'].values
    trainX, trainY = train['tweet'].values, train['polarity'].values

    trainY = np.where(trainY == 4, 1, trainY)
    testY = np.where(testY == 4, 1, testY)

    # preprocessor = Preprocessor()

    # trainX, testX = preprocessor.clean_texts(trainX, testX)

    empty_tweets = []

    for index, tweet in enumerate(trainX):
        if len(tweet) == 0:
            empty_tweets.append(index)

    trainX = np.delete(trainX, empty_tweets)
    trainY = np.delete(trainY, empty_tweets)

    empty_tweets = []

    for index, tweet in enumerate(testX):
        if len(tweet) == 0:
            print(tweet)
            empty_tweets.append(index)

    testX = np.delete(testX, empty_tweets)
    testY = np.delete(testY, empty_tweets)

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(trainX)
    
    vocab_size = int(len(tokenizer.word_index)) + 1

    tokenizer.num_words = vocab_size

    encoded_train = []

    for tweet in trainX:
        encoded_train.append(one_hot(tweet, vocab_size))
    
    trainX = pad_sequences(encoded_train, maxlen=max_len, padding='post')

    tokenizer.fit_on_texts(testX)

    encoded_test = []

    for tweet in testX:
        encoded_test.append(one_hot(tweet, vocab_size))

    testX = pad_sequences(encoded_test, maxlen=max_len, padding='post')

    return trainX, trainY, testX, testY, vocab_size, max_len

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
    
    trainX, trainY, testX, testY, vocab_size, max_len = prepare_data()

    # file = open('data.pkl', 'rb')
    # trainX = pickle.load(file)
    # trainY = pickle.load(file)
    # testX = pickle.load(file)
    # testY = pickle.load(file)
    # vocab_size = pickle.load(file)
    # max_len = 140
    # file.close()

    # file = open('data.pkl','wb')
    # pickle.dump(trainX, file)
    # pickle.dump(trainY, file)
    # pickle.dump(testX, file)
    # pickle.dump(testY, file)
    # pickle.dump(vocab_size, file)
    # pickle.dump(max_len, file)
    # file.close()

    neural_network = NeuralNetwork()

    neural_network.create_model(vocab_size, max_len)

    neural_network.model.summary()
    
    neural_network.model.compile(optimizer=tf.train.AdamOptimizer(),
            loss='binary_crossentropy',
            metrics=['accuracy'])
   
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

    #plot_graph(history)

    write_csv(create_csv_dataframe(history, results))

if __name__ == '__main__':
    for i in range(CONFIG.getint('DEFAULT', 'RUNS_PER_EPOCH')):
        main()