import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

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

    flatten = keras.layers.Flatten()

    encodeModelLayers = [encodeInputLayer]
    encodeModelLayers.extend(hiddenLayers)
    encodeModelLayers.append(flatten)
    encodeModelLayers.append(keras.layers.Activation(activation=tf.nn.sigmoid))

    encodeModel = keras.Sequential(encodeModelLayers)
    encodeModel.compile(optimizer="rmsprop", loss='mean_squared_error')

    retargetSize = (int(inputShape[0]/(2 ** len(filterSizes))), int(inputShape[1] / (2 ** len(filterSizes))), numberOfFilters[-1])
    flattenedLen = retargetSize[0] * retargetSize[1] * retargetSize[2]
    decodeReshapeLayer  = keras.layers.Reshape(target_shape=retargetSize, input_shape=(flattenedLen,),  name="Decode_reshape")

    hiddenDecodeLayers = []
    i = 0
    while i < len(hiddenLayers):
        hiddenDecodeLayers.append(keras.layers.UpSampling2D((2, 2)))
        i = i + 1
        if useDroput:
            i = i + 1
            hiddenDecodeLayers.append(keras.layers.Dropout(rate=0.5))
        if i == len(hiddenLayers):
            break
        hiddenDecodeLayers.append(hiddenLayers[-(i+1)])
        i = i+1

    decodeOutputLayer = keras.layers.Conv2D(filters=inputShape[2], kernel_size=(2, 2), strides=(1, 1), activation=tf.nn.sigmoid, padding='same')

    decodeModelLayers = [decodeReshapeLayer]
    decodeModelLayers.extend(hiddenDecodeLayers)
    decodeModelLayers.append(decodeOutputLayer)

    decodeModel = keras.Sequential(decodeModelLayers)
    decodeModel.compile(optimizer="rmsprop", loss='mean_squared_error')

    return encodeModel, decodeModel


def trainCNAE(encodeModel, decodeModel, dataSet, iterations):
    encodedLayers = encodeModel.layers
    decodedLayers = decodeModel.layers
    train = []
    train.extend(encodedLayers)
    train.extend(decodedLayers)
    trainModel = keras.Sequential(train)
    trainModel.compile(optimizer='rmsprop', loss='mean_squared_error')
    trainModel.fit(dataSet, dataSet, epochs=iterations)


def testCNAE(encodeModel, decodeModel, dataSet):
    encoded = encodeModel.predict(dataSet)
    decoded = decodeModel.predict(encoded)
    test_loss = decodeModel.evaluate(encoded, dataSet)
    print(test_loss)
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 10, i*2 + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(dataSet[i], cmap=plt.cm.hsv)
        plt.xlabel("Original")
        plt.subplot(5, 10, i * 2 + 2)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(decoded[i], cmap=plt.cm.hsv)
        plt.xlabel("Reconstructed")
    plt.show()
    return test_loss


(trainImg, _), (testImg, _) = keras.datasets.cifar100.load_data()
trainImg = trainImg/256
testImg = testImg/256
enc, dec = generateNetworks((32, 32, 3), [64], [(4, 4)], useDroput=True)
trainCNAE(enc, dec, trainImg, 5)
testCNAE(enc, dec, testImg)