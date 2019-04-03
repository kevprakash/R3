import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def generateNetworks(inputShape, numberOfFilters, filterSizes, useDroput=True):
    assert len(numberOfFilters) == len(filterSizes)
    encodeInputLayer = keras.layers.Conv2D(input_shape=inputShape, filters=numberOfFilters[0], kernel_size=filterSizes[0], activation=tf.nn.relu, padding='same', name="Input")
    hiddenLayers = [keras.layers.MaxPool2D((2, 2), padding='same')]
    if useDroput:
        hiddenLayers.insert(0, keras.layers.Dropout(rate=0.5))
    for i in range(len(filterSizes) - 1):
        hidden = keras.layers.Conv2D(filters=numberOfFilters[i+1], kernel_size=filterSizes[i+1], activation=tf.nn.relu, padding='same', name="%s%d"%("Hidden_", (i+1)))
        hiddenLayers.append(hidden)
        if useDroput:
            hiddenLayers.append(keras.layers.Dropout(rate=0.5))
        hiddenLayers.append(keras.layers.MaxPool2D((2, 2), padding='same'))

    encodeModelLayers = [encodeInputLayer]
    encodeModelLayers.extend(hiddenLayers)
    encodeModelLayers.append(keras.layers.Activation(activation=tf.nn.relu))

    encodeModel = keras.Sequential(encodeModelLayers)
    encodeModel.compile(optimizer="rmsprop", loss='mean_squared_error')

    _, r1, r2, r3 = encodeModel.output_shape
    retargetSize = (r1, r2, r3)

    hiddenDecodeLayers = []
    i = 0
    while i < len(hiddenLayers):
        if (i == 0):
            hiddenDecodeLayers.append(keras.layers.UpSampling2D((2, 2), input_shape=retargetSize, name="%s%d" % ("Upsampling", i + 1)))
        else:
            hiddenDecodeLayers.append(keras.layers.UpSampling2D((2, 2), name="%s%d"%("Upsampling", i+1)))
        i = i + 1
        if useDroput:
            i = i + 1
            hiddenDecodeLayers.append(keras.layers.Dropout(rate=0.5))
        if i == len(hiddenLayers):
            break
        hiddenDecodeLayers.append(hiddenLayers[-(i+1)])
        i = i+1

    decodeOutputLayer = keras.layers.Conv2D(filters=inputShape[2], kernel_size=(2, 2), strides=(1, 1), activation=tf.nn.sigmoid, padding='same')

    decodeModelLayers = []
    decodeModelLayers.extend(hiddenDecodeLayers)
    decodeModelLayers.append(decodeOutputLayer)

    decodeModel = keras.Sequential(decodeModelLayers)
    decodeModel.compile(optimizer="rmsprop", loss='mean_squared_error')

    return encodeModel, decodeModel


def trainCNAE(encodeModel, decodeModel, dataSet, batchSize, iterations):
    encodedLayers = encodeModel.layers
    decodedLayers = decodeModel.layers
    train = []
    train.extend(encodedLayers)
    train.extend(decodedLayers)
    trainModel = keras.Sequential(train)
    trainModel.compile(optimizer='rmsprop', loss='mean_squared_error')
    #for layer in trainModel.layers:
        #print(layer.output_shape)
    trainModel.fit(dataSet, dataSet, batch_size=batchSize, epochs=iterations)


def testCNAE(encodeModel, decodeModel, dataSet, displayedRows=5, displayedColumns=5, batchSize=1):
    encoded = encodeModel.predict(dataSet, batch_size=batchSize)
    decoded = decodeModel.predict(encoded, batch_size=batchSize)
    test_loss = decodeModel.evaluate(encoded, dataSet, batch_size=batchSize)
    print(test_loss)
    plt.figure(figsize=(19, 11))
    for i in range(min(displayedRows * displayedColumns, len(dataSet))):
        plt.subplot(displayedRows, displayedColumns * 2, i*2 + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        inImage = np.reshape(dataSet[i], newshape=(len(dataSet[i][0]), len(dataSet[i]), len(dataSet[i][0][0])))
        plt.imshow(inImage, cmap=plt.cm.hsv)
        plt.xlabel("Original")
        plt.subplot(displayedRows, displayedColumns * 2, i * 2 + 2)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        outImage = np.reshape(decoded[i], newshape=(len(decoded[i][0]), len(decoded[i]), len(decoded[i][0][0])))
        plt.imshow(outImage, cmap=plt.cm.hsv)
        plt.xlabel("Reconstructed")
    plt.show()
    return test_loss