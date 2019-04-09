import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from config import CONFIG
import json
from keras.models import model_from_json

class NeuralNetwork:

    checkpoint_filepath = 'weights.best.hdf5'

    def create_model(self, vocab_size, max_len):

        # # load saved model architecture
        # json_file = open('model.json', 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()

        # loaded_model = tf.keras.models.model_from_json(loaded_model_json)

        # # load saved model weights
        # loaded_model.load_weights('model.h5')

        # embedding_layer = loaded_model.layers[0]

        # embedding_layer.trainable = False

        self.model = keras.Sequential()
        self.model.add(keras.layers.Embedding(vocab_size, CONFIG.getint('DEFAULT', 'EMBEDDING_OUTPUT'), 
        input_length=max_len))
        #self.model.add(embedding_layer)
        self.model.add(keras.layers.GlobalAveragePooling1D())
        #self.model.add(keras.layers.Flatten())
        #self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(CONFIG.getint('DEFAULT', 'HIDDEN'), activation=tf.nn.relu))
        #self.model.add(keras.layers.Dropout(0.5))
        # self.model.add(keras.layers.Dense(32, activation=tf.nn.relu))
        # self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(CONFIG.getint('DEFAULT', 'OUTPUT'), activation=tf.nn.softmax))

    def fit_model(self, partial_x_train, partial_y_train, x_val, y_val):
        return self.model.fit(partial_x_train, partial_y_train, 
                        epochs=CONFIG.getint('DEFAULT', 'EPOCHS'), 
                        batch_size=CONFIG.getint('DEFAULT', 'BATCH_SIZE'), 
                        validation_data=(x_val, y_val), verbose=1,
                        callbacks=self.get_callbacks())

    def get_callbacks(self):
        
        callback_list = []

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                    patience=CONFIG.getint('DEFAULT', 'PATIENCE'),
                                    verbose=1,
                                    mode='min')

        checkpoint = keras.callbacks.ModelCheckpoint(self.checkpoint_filepath,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True,
                                mode='max')

        callback_list.append(early_stopping)
        callback_list.append(checkpoint)

        return callback_list
