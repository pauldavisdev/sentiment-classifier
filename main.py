from csv_handler import read_csv
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, one_hot
from neural_network import NeuralNetwork
from preprocessor import Preprocessor
import matplotlib.pyplot as plt
import time, os
import pandas as pd

def main():
    num_csv_rows = 100000

    max_len = 140

    vocab_size = 1000

    train, test = read_csv(num_csv_rows)

    testX, testY = test['tweet'].values, test['polarity'].values
    trainX, trainY = train['tweet'].values, train['polarity'].values

    trainY = np.where(trainY == 4, 1, trainY)
    testY = np.where(testY == 4, 1, testY)

    preprocessor = Preprocessor()

    trainX, testX = preprocessor.clean_texts(trainX, testX)

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(trainX)

    tokenizer.num_words = vocab_size
    
    encoded_train = []

    for tweet in trainX:
        encoded_train.append(one_hot(tweet, vocab_size))
    
    #wait = input("PRESS ENTER TO CONTINUE.")

    trainX = pad_sequences(encoded_train, maxlen=max_len, padding='post')

    tokenizer.fit_on_texts(testX)

    encoded_test = []

    for tweet in testX:
        encoded_test.append(one_hot(tweet, vocab_size))

    testX = pad_sequences(encoded_test, maxlen=max_len, padding='post')

    neural_network = NeuralNetwork()

    neural_network.generate_model(vocab_size)

    neural_network.model.summary()

    neural_network.model.compile(optimizer=tf.train.AdamOptimizer(),
            loss='binary_crossentropy',
            metrics=['accuracy'])

    x_val = trainX[:5000]
    partial_x_train = trainX[5000:]

    y_val = trainY[:5000]
    partial_y_train = trainY[5000:]

    history = neural_network.fit_model(partial_x_train, partial_y_train, 
                                        x_val, y_val)

    results = neural_network.model.evaluate(testX, testY)

    print(neural_network.model.metrics_names)
    print(results)

    history_dict = history.history
    history_dict.keys()

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()   # clear figure
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    timestamp = int(time.time())

    df = pd.DataFrame.from_dict(history_dict)
    df['test_loss'] = results[0]
    df['test_acc'] = results[1]
    df['input nodes'] = neural_network.num_input
    df['hidden nodes'] = neural_network.num_hidden
    df['output nodes'] = neural_network.num_output
    df['epochs'] = neural_network.epochs
    df['batch size'] = neural_network.batch_size
    df['patience'] = neural_network.patience
    df = df.round(decimals=4)
    df['timestamp'] = timestamp

    print(df)

    log_path = 'logs/log.csv'

    if os.path.exists(log_path):
        copyfile(log_path, ''.join(['logs/log', '_', str(timestamp), '.csv']))
        # append to csv
        with open(log_path, 'a') as f:
            df.to_csv(f, header=True, index=False)
    else:
        # create new csv
        df.to_csv(path_or_buf=log_path, index=False)

if __name__ == '__main__':
    main()