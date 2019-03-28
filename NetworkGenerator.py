import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow import keras


def initializeNetwork(inputShape, nodesPerLayer, outputLength, activation, useDropout, optimizer='rmsprop'):
    "Generates LSTM networks given hyperparameters"
    network = keras.Sequential()
    network.add(keras.layers.LSTM(nodesPerLayer[0], input_shape=inputShape, name='InputLayer', return_sequences=True))
    for x in range(1, len(nodesPerLayer)-1):
        network.add(keras.layers.LSTM(nodesPerLayer[x], return_sequences=True))
        if useDropout:
            network.add(keras.layers.Dropout(rate=0.5))
    network.add(keras.layers.LSTM(nodesPerLayer[-1], return_sequences=False))
    if useDropout:
        network.add(keras.layers.Dropout(rate=0.5))
    network.add(keras.layers.Dense(outputLength, activation=activation))
    network.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return network


def generateTrainingData(numOfInputs=1000, maxLength=5, verbose=False):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    charToInt = dict((c, i+1) for i, c in enumerate(alphabet))
    dataX = []
    dataY = []
    for i in range(numOfInputs):
        start = np.random.randint(len(alphabet) - 2)
        end = np.random.randint(start, min(start + maxLength, len(alphabet) - 1))
        sequence_in = alphabet[start:end + 1]
        sequence_out = alphabet[end + 1]
        dataX.append([charToInt[char] for char in sequence_in])
        dataY.append(charToInt[sequence_out])
        if verbose:
            print(sequence_in, '->', sequence_out)
    x = pad_sequences(dataX, maxlen=maxLength, dtype='float32')
    x = np.reshape(x, (x.shape[0], maxLength, 1))

    return x, dataY


def initializeNetworkTest(inputModel, numOfInputs=1000, maxLength=5, iterations=100, epochs=5, batchSize=1, verbose=False):
    for rep in range(iterations):
        x, y = generateTrainingData(numOfInputs, maxLength, verbose)
        inputModel.fit(x=x, y=y, verbose=verbose, batch_size=batchSize, epochs=epochs)
        print('%g%s'%((rep+1)/iterations * 100, "%"))

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    intToChar = dict((i + 1, c) for i, c in enumerate(alphabet))

    x2, y2 = generateTrainingData(20, maxLength, verbose=verbose)

    prediction = model.predict(x2, verbose=0)
    for i in range(len(x2)):
        index = np.argmax(prediction[i])
        result = intToChar[index]
        seqIn = []
        for j in range(len(x2[i])):
            if x2[i][j][0] > 0:
                seqIn.append(intToChar[x2[i][j][0]])
        print(seqIn, "->", result)


model = initializeNetwork((5, 1,), [32], 26, 'softmax', False)
initializeNetworkTest(model, numOfInputs=100)
