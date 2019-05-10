from keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from tensorflow import keras
import R3Utilities as R3Util

def initializeControllerNetwork(inputModel, nodesPerLayer, outputLength, hiddenActivation=tf.nn.relu, dropoutRate=0, optimizer=tf.keras.optimizers.Adam, recurrentCell=keras.layers.SimpleRNN, learningRate=0.1):
    "Generates Recurrent networks given hyperparameters"
    initializer = tf.initializers.random_normal #makes weights random when you make a layer
    network = keras.Sequential() #each layer in the model will have a unique layer to pass their info into
    network.add(keras.layers.TimeDistributed(inputModel, input_shape=(None, inputModel.input_shape[1], inputModel.input_shape[2], inputModel.input_shape[3])))
    for x in range(0, len(nodesPerLayer)):
        network.add(recurrentCell(nodesPerLayer[x], return_sequences=True, activation=hiddenActivation, recurrent_initializer=initializer, kernel_initializer=initializer))
        if dropoutRate > 0:
            network.add(keras.layers.Dropout(rate=dropoutRate))
    network.add(recurrentCell(outputLength, return_sequences=False, activation=tf.nn.softmax, recurrent_initializer=initializer, kernel_initializer=initializer))
    network.compile(optimizer=optimizer(lr=learningRate), loss='sparse_categorical_crossentropy')

    #for layer in network.layers:
    #    layer.trainable = False

    #for layer in inputModel.layers:
    #   layer.trainable = False

    return network