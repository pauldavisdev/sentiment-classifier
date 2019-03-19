import tensorflow as tf
from tensorflow import keras

class NeuralNetwork:

    num_input = 48
    num_hidden = 24
    num_output = 1
    epochs = 10
    batch_size = 128

    def generate_model(self, vocab_size):
        self.model = keras.Sequential()
        self.model.add(keras.layers.Embedding(vocab_size, self.num_input))
        self.model.add(keras.layers.GlobalAveragePooling1D())
        self.model.add(keras.layers.Dense(self.num_hidden, activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(self.num_output, activation=tf.nn.sigmoid))

    def fit_model(self, partial_x_train, partial_y_train, x_val, y_val):
        return self.model.fit(partial_x_train, partial_y_train, 
                        epochs=self.epochs, batch_size=self.batch_size, 
                        validation_data=(x_val, y_val), verbose=1)
