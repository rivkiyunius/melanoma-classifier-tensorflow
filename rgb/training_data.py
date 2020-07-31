import time
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
import pickle
import os
import datetime

# log_dir = os.path.join(
#     "logs",
#     datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
# )

gpu_option = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_option))

X_train = pickle.load(open("X_train.pickle", "rb"))
y_train = pickle.load(open("y_train.pickle", "rb"))
X_test = pickle.load(open("X_test.pickle", "rb"))
y_test = pickle.load(open("y_test.pickle", "rb"))

X_train = np.divide(X_train, 255.0)

X_test = np.divide(X_test, 255.0)

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            log_dir = os.path.join(
                "coba",
                NAME,
            )
            tensorboard = TensorBoard(log_dir=log_dir)
            model = Sequential()
            # proses convolution
            model.add(Conv2D(layer_size, (5, 5), input_shape=X_train.shape[1:]))
            model.add(Activation('relu'))
            # proses pooling layer
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer - 1):
                model.add(Conv2D(layer_size, (5, 5)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            # proses fully conected layer
            model.add(Flatten())
            for l in range(dense_layer):
                model.add(Dense(dense_layer))
                model.add(Activation('relu'))
                model.add(Dropout(0.5))
                model.add(Activation('softmax'))
                model.add(Dropout(0.5))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

            model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test), callbacks=[tensorboard])

            # model.save("result.model")
