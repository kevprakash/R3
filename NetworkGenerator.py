import tensorflow as tf
import random
import numpy as np
from tensorflow import keras


def initializeNetwork(inputShape, nodesPerLayer, outputLength, activation, useDropout):
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
    network.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return network


def generateTrainingData(maxVal, inputSize, size):
    "Generate Training Data and labels"
    temp = []
    for x in range(0, inputSize):
        temp.append(0)
    data = [temp]
    combined = [0]
    for y in range(1, size):
        temp2 = []
        for z in range(0, inputSize):
            temp2.append(random.randint(0, maxVal))
        data.append(temp2)
        combined.append(np.sum(temp2) + int(combined[y - 1] / 2))
    data = tf.reshape(data, shape=(size, 1, inputSize))
    combined = tf.reshape(combined, shape=(size, 1))
    return data, combined


def initializeNetworkTest(inputModel, maxValue = 10, inSize = 10, sequenceLength = 100, batchSize = 1, iterations = 500):
    sess = tf.Session()
    for i in range(0, iterations):
        trainData, trainLabel = generateTrainingData(maxValue, inSize, sequenceLength)
        model.fit(x=trainData, y=trainLabel, verbose=False, batch_size=batchSize, steps_per_epoch=int(sequenceLength/batchSize), epochs=1)
        print("%g%s" % (i/iterations*100, "%"))

    testData, testLabel = generateTrainingData(maxValue, inSize, sequenceLength)
    pred = model.predict(testData, batch_size=batchSize, steps=int(sequenceLength/batchSize))
    model.evaluate(testData, testLabel, verbose=True, steps=int(sequenceLength/batchSize))

    print()
    for j in range(0, 10):
        print(sess.run(testData[j]), ":", sess.run(testLabel[j]), " -> ", np.argmax(pred[j]))
        print()


inLen = 10
mVal = 10
model = initializeNetwork((1, inLen,), [32], inLen * mVal * 2 + 1, 'softmax', True)
initializeNetworkTest(model, mVal, inLen)
