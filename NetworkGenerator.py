import tensorflow as tf
from tensorflow import keras


def initializeNetwork(inputShape, nodesPerLayer):
    "Generates LSTM networks given hyperparameters"
    model = keras.Sequential()
    model.add(keras.layers.LSTM(nodesPerLayer[0], input_shape=inputShape))
    for i in range(1,  len(nodesPerLayer) - 1):
        model.add(keras.layers.LSTM(nodesPerLayer[i]))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def loadNetwork(inputShape, nodesPerLayer, parameters):
    "Generates LSTM networks given paramters"
    model = keras.Sequential()
    inputLayer = keras.layers.LSTM(nodesPerLayer[0], input_shape=inputShape)
    inputLayer.set_weights(parameters[0])
    model.add(inputLayer)
    for i in range(1, len(nodesPerLayer) - 1):
        layer = keras.layers.LSTM(nodesPerLayer[i])
        layer.set_weights(parameters[i])
        model.add(layer)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = initializeNetwork((10, 1), [10])

for x in model.get_weights():
    print(x.shape)
    print()
print(model.output_shape)

#model2 = loadNetwork((10, 1), [10, 10], model.get_weights())

#print(model.get_weights().__eq__(model2.get_weights()))
