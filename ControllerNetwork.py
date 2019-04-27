from keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from tensorflow import keras
import R3Utilities as R3Util

def initializeControllerNetwork(inputModel, nodesPerLayer, latentSpaceLength, outputLength, hiddenActivation=tf.nn.relu, activation=tf.nn.sigmoid, dropoutRate=0, optimizer='rmsprop'):
    "Generates LSTM networks given hyperparameters"
    initializer = tf.initializers.random_normal
    network = keras.Sequential()
    inputModelLayers = inputModel.layers
    for l in inputModelLayers:
        network.add(l)
    network.add(keras.layers.Reshape((1, latentSpaceLength)))
    for x in range(0, len(nodesPerLayer) - 1):
        network.add(keras.layers.LSTM(nodesPerLayer[x], return_sequences=True, activation=hiddenActivation, recurrent_initializer=initializer, kernel_initializer=initializer))
        print(x)
        if dropoutRate > 0:
            network.add(keras.layers.Dropout(rate=dropoutRate))
    network.add(keras.layers.LSTM(nodesPerLayer[-1], return_sequences=False, activation=hiddenActivation, recurrent_initializer=initializer, kernel_initializer=initializer))
    if dropoutRate > 0:
        network.add(keras.layers.Dropout(rate=dropoutRate))
    network.add(keras.layers.Dense(outputLength, activation=activation, kernel_initializer=initializer))
    network.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return network


def generateTrainingData(numOfInputs=1000, maxLength=5, verbose=False):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    charToInt = dict((c, i+1) for i, c in enumerate(alphabet))
    dataX = []
    dataY = []
    for i in range(numOfInputs):
        start = np.random.randint(len(alphabet) - 2)
        end = np.random.randint(start, min(start + maxLength, len(alphabet)-1))
        sequence_in = alphabet[start:(end+1)]
        sequence_out = alphabet[end + 1]
        dataX.append([charToInt[char] for char in sequence_in])
        dataY.append(charToInt[sequence_out])
        if verbose:
            print(sequence_in, '->', sequence_out)
    x = pad_sequences(dataX, maxlen=maxLength, dtype='float32')
    x = np.reshape(x, (x.shape[0], maxLength, 1))

    return x, dataY


def initializeControllerNetworkTest(inputModel, numOfInputs=1000, maxLength=5, iterations=100, epochs=5, batchSize=1, verbose=False):
    for rep in range(iterations):
        x, y = generateTrainingData(numOfInputs, maxLength, verbose)
        inputModel.fit(x=x, y=y, verbose=verbose, batch_size=batchSize, epochs=epochs)
        if rep == 0:
            print("Training start")
        R3Util.printLoadBar(rep/iterations, 50)
    R3Util.printLoadBar(1, 50)
    print("Training end")

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    intToChar = dict((i+1, c) for i, c in enumerate(alphabet))

    x2, y2 = generateTrainingData(20, maxLength, verbose=verbose)

    prediction = inputModel.predict(x2, verbose=0)
    for i in range(len(x2)):
        index = np.argmax(prediction[i])
        result = intToChar[index]
        seqIn = []
        for j in range(len(x2[i])):
            if x2[i][j][0] > 0:
                seqIn.append(intToChar[x2[i][j][0]])
        print(seqIn, "->", result, ":", index)