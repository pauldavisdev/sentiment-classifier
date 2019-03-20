import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping

class NeuralNetwork:

    num_input = 500
    num_hidden = 350
    num_output = 1
    epochs = 50
    batch_size = 64
    patience = 3

    def create_model(self, vocab_size):
        self.model = keras.Sequential()
        self.model.add(keras.layers.Embedding(vocab_size, self.num_input))
        self.model.add(keras.layers.GlobalAveragePooling1D())
        self.model.add(keras.layers.Dense(self.num_hidden, activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(self.num_output, activation=tf.nn.sigmoid))

    def fit_model(self, partial_x_train, partial_y_train, x_val, y_val):
        return self.model.fit(partial_x_train, partial_y_train, 
                        epochs=self.epochs, batch_size=self.batch_size, 
                        validation_data=(x_val, y_val), verbose=1,
                        callbacks=self.get_callbacks())

    def get_callbacks(self):
        
        callback_list = []

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                    patience=self.patience,
                                    verbose=1,
                                    mode='min')

        callback_list.append(early_stopping)

        return callback_list
