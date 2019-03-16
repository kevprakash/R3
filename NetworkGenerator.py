from tensorflow import keras


def initializeNetwork(inputShape, nodesPerLayer):
    "Generates networks given hyperparameters"
    layers = [keras.layers.LSTM(nodesPerLayer[0], input_shape=inputShape)]
    for i in range(1,  nodesPerLayer.len() - 1):
        layers.append(keras.layers.LSTM(nodesPerLayer[i]))
    return keras.Sequential(layers)


def initializeNetwork(inputShape, nodesPerLayer, parameters):
    "Generates networks given paramters"
    inputLayer = keras.layers.LSTM(nodesPerLayer[0], input_shape=inputShape)
    inputLayer.set_weights(parameters[0])
    layers = [inputLayer]
    for i in range(1, nodesPerLayer.len() - 1):
        layer = keras.layers.LSTM(nodesPerLayer[i])
        layer.set_weights(parameters[i])
        layers.append(layer)
    return keras.Sequential(layers)