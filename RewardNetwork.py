from tensorflow import keras

def initializeRewardNetwork(convolutionNetwork, controllerNetwork, nodesPerLayer, outputLength, dropoutRate=0, optimizer='rmsprop', hiddenActivation='relu', activation='sigmoid'):
    x = keras.layers.Concatenate([convolutionNetwork, controllerNetwork], axis=1)
    for i in range(0, len(nodesPerLayer) - 1):
        x = keras.layers.LSTM(nodesPerLayer[i], return_sequences=True, activation=hiddenActivation)(x)
        if dropoutRate > 0:
            x = keras.layers.Dropout(rate=dropoutRate)(x)
    x = keras.layers.LSTM(nodesPerLayer[-1], return_sequences=False, activation=hiddenActivation)(x)
    output = keras.layers.Dense(outputLength, activation=activation)(x)
    model = keras.Model(inputs=[convolutionNetwork, controllerNetwork], outputs=output)

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model