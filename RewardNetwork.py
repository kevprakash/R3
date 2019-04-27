from tensorflow import keras

def initializeRewardNetwork(convolutionNetwork, controllerNetwork, latentSpaceLength, controllerOutputLength, nodesPerLayer, outputLength, lossFunction, dropoutRate=0, optimizer='rmsprop', hiddenActivation='relu', activation='sigmoid'):

    model = keras.Sequential()
    model.add(keras.layers.Concatenate([convolutionNetwork.output, controllerNetwork.output]))
    model.add(keras.layers.Reshape((1, latentSpaceLength + controllerOutputLength)))
    for i in range(0, len(nodesPerLayer) - 1):
        model.add(keras.layers.LSTM(nodesPerLayer[i], return_sequences=True, activation=hiddenActivation))
        if dropoutRate > 0:
            model.add(keras.layers.Dropout(rate=dropoutRate))
    model.add(keras.layers.LSTM(nodesPerLayer[-1], return_sequences=False, activation=hiddenActivation))
    model.add(keras.layers.Dense(outputLength, activation=activation))

    model.compile(optimizer=optimizer, loss=lossFunction)

    return model