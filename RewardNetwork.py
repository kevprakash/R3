import  tensorflow as tf
from tensorflow import keras

def initializeRewardNetwork(convolutionNetwork, controllerNetwork, latentSpaceLength, controllerOutputLength, nodesPerLayer, outputLength, dropoutRate=0, optimizer='rmsprop', hiddenActivation='relu', activation='sigmoid'):
    initializer = tf.initializers.random_normal
    model = keras.Sequential()
    model.add(keras.layers.Concatenate([convolutionNetwork.output, controllerNetwork.output]))

    sharedLayers = []
    sharedLayers.append(keras.layers.Reshape((1, latentSpaceLength + controllerOutputLength)))
    for i in range(0, len(nodesPerLayer) - 1):
        sharedLayers.append(keras.layers.LSTM(nodesPerLayer[i], return_sequences=True, activation=hiddenActivation, recurrent_initializer=initializer, kernel_initializer=initializer))
        if dropoutRate > 0:
            sharedLayers.append(keras.layers.Dropout(rate=dropoutRate))
    sharedLayers.append(keras.layers.LSTM(nodesPerLayer[-1], return_sequences=False, activation=hiddenActivation, recurrent_initializer=initializer, kernel_initializer=initializer))
    sharedLayers.append(keras.layers.Dense(outputLength, activation=activation, kernel_initializer=initializer))

    for layer in sharedLayers:
        model.add(layer)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    inputLayer = keras.layers.Input(shape=(controllerOutputLength,), dtype='float32')
    x = keras.layers.concatenate([convolutionNetwork.output, inputLayer])
    for layer in sharedLayers:
        x = layer(x)
    trainer = keras.Model(inputs=[convolutionNetwork.input, inputLayer], outputs=x)

    return model, trainer

