import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import random


def imageLoss(img1, img2):
    return tf.reduce_mean(1-tf.image.ssim_multiscale(img1=img1, img2=img2, max_val=1.0))


def generateNetworks(inputShape, numberOfFilters, filterSizes, latentSpaceLength, hiddenActivation=tf.nn.relu, learningRate=0.0001, dropoutRate=0.1):
    assert len(numberOfFilters) == len(filterSizes)
    inputLayer = keras.layers.Conv2D(input_shape=inputShape, filters=numberOfFilters[0], kernel_size=filterSizes[0], padding='same', activation=hiddenActivation, name='EncodeInput')
    encodeLayers = [inputLayer]
    for i in range(1, len(filterSizes)):
        encodeLayers.append(keras.layers.MaxPool2D(pool_size=filterSizes[i-1], padding='same'))
        if dropoutRate > 0:
            encodeLayers.append(keras.layers.Dropout(rate=dropoutRate))
        encodeLayers.append(keras.layers.Conv2D(filters=numberOfFilters[i], kernel_size=filterSizes[i], padding='same', activation=hiddenActivation, name="%s%d" % ("Convolution_", i)))
    encodeLayers.append(keras.layers.Flatten())
    encodeLayers.append(keras.layers.Dense(latentSpaceLength, activation=tf.nn.sigmoid, name="EncodeOutput"))

    encode = keras.Sequential(encodeLayers)
    encode.compile(optimizer=tf.keras.optimizers.Adam(lr=learningRate), loss='mean_squared_error')

    _, d1, d2, d3 = encode.layers[-3].output_shape
    reshape = (d1, d2, d3)
    reshapeLength = d1 * d2 * d3

    _, o2 = encode.output_shape

    decodeInput = keras.layers.Dense(input_shape=(o2,), units=reshapeLength, name="DecodeInput")
    decodeReshape = keras.layers.Reshape(target_shape=reshape, name='Reshape')
    decodeLayers = [decodeInput, decodeReshape]
    if dropoutRate > 0:
        decodeLayers.append(keras.layers.Dropout(rate=dropoutRate))
    for i in range(len(filterSizes)-1, 0, -1):
        decodeLayers.append(keras.layers.Conv2DTranspose(filters=numberOfFilters[i-1], strides= filterSizes[i-1], kernel_size=filterSizes[i-1], padding='same', activation=hiddenActivation, name="%s%d" % ("Deconvolution_", i)))
        #decodeLayers.append(keras.layers.UpSampling2D(size=filterSizes[i-1]))
        if dropoutRate > 0:
            decodeLayers.append(keras.layers.Dropout(rate=dropoutRate))
    decodeLayers.append(keras.layers.Conv2DTranspose(filters=inputShape[2], kernel_size=(2, 2), padding='same', activation=tf.nn.sigmoid, name='DecodeOutput'))

    decode = keras.Sequential(decodeLayers)
    decode.compile(optimizer=tf.keras.optimizers.Adam(lr=learningRate), loss=imageLoss)

    combinedLayers = []
    combinedLayers.extend(encodeLayers)
    combinedLayers.extend(decode.layers)
    combined = keras.Sequential(combinedLayers)
    combined.compile(optimizer=tf.keras.optimizers.Adam(lr=learningRate), loss=imageLoss)

    return encode, decode, combined


def encodeNetwork(encoderModel):
    _, s1, s2, s3 = encoderModel.input_shape
    inputShape = (s1, s2, s3)
    numberOfFilters = []
    filterSizes = []
    _, latentSpaceLength = encoderModel.output_shape
    useDropout=False
    hiddenActivation=None
    learningRate = keras.backend.eval(encoderModel.optimizer.lr)
    for layer in encoderModel.layers:
        if isinstance(layer, keras.layers.Conv2D):
            numberOfFilters.append(layer.output_shape[3])
            filterSizes.append((np.shape(layer.get_weights()[0])[0], np.shape(layer.get_weights()[0])[1]))
        elif isinstance(layer, keras.layers.Dropout):
            useDropout=True
    possibleActivations = [tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid, tf.nn.leaky_relu]
    hiddenActivation = possibleActivations[random.randint(0, len(possibleActivations) - 1)]
    return inputShape, numberOfFilters, filterSizes, latentSpaceLength, hiddenActivation, learningRate, useDropout


def trainCNAE(model, dataSet, batchSize, iterations, verbose=True):
    model.fit(dataSet, dataSet, batch_size=batchSize, epochs=iterations, verbose=verbose)


def testCNAE(encodeModel, decodeModel, dataSet, displayedRows=5, displayedColumns=5, batchSize=1):
    encoded = encodeModel.predict(dataSet, batch_size=batchSize)
    decoded = decodeModel.predict(encoded, batch_size=batchSize)
    test_loss = decodeModel.evaluate(encoded, dataSet, batch_size=batchSize)
    print(test_loss)
    plt.figure(figsize=(19, 11))
    for i in range(0, (displayedRows * displayedColumns)):
        x = random.randint(0, len(dataSet)-1)
        plt.subplot(displayedRows, displayedColumns * 2, i * 2 + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        inImage = np.reshape(dataSet[x], newshape=(len(dataSet[x][0]), len(dataSet[x]), len(dataSet[x][0][0])))
        plt.imshow(inImage, cmap=plt.cm.hsv)
        plt.xlabel("Original")
        plt.subplot(displayedRows, displayedColumns * 2, i * 2 + 2)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        outImage = np.reshape(decoded[x], newshape=(len(decoded[x][0]), len(decoded[x]), len(decoded[x][0][0])))
        plt.imshow(outImage, cmap=plt.cm.hsv)
        plt.xlabel("Reconstructed")
    plt.show()
    return test_loss