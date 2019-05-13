import tensorflow as tf
from tensorflow import keras

def generateNetwork(imageSize, numberOfFilters, filterSizes, strideSizes, hiddenNodes, outputLength, optimizer=keras.optimizers.SGD, learningRate=0.01, recurrentActivations=tf.nn.sigmoid, convolutionActivations=tf.nn.tanh, hiddenActivations=tf.nn.tanh):
    print("Generating Model")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    convolutionInput = keras.layers.Input(shape=(None, imageSize[1], imageSize[0], 3), name="Convolution_Input")
    x = keras.layers.TimeDistributed(keras.layers.Activation(activation=None))(convolutionInput)
    for i in range(len(numberOfFilters)):
        x = keras.layers.ConvLSTM2D(filters=numberOfFilters[i], kernel_size=filterSizes[i], strides=strideSizes[i], padding='same', activation=convolutionActivations, recurrent_activation=recurrentActivations, kernel_initializer=initializer, recurrent_initializer=initializer, bias_initializer=initializer, return_sequences=True)(x)
    x = keras.layers.TimeDistributed(keras.layers.Flatten())(x)
    for i in range(len(hiddenNodes)):
        x = keras.layers.LSTM(hiddenNodes[i], activation=hiddenActivations, recurrent_activation=recurrentActivations, kernel_initializer=initializer, bias_initializer=initializer, recurrent_initializer=initializer, return_sequences=True)(x)
    output = keras.layers.LSTM(outputLength, activation=None, recurrent_activation=recurrentActivations, kernel_initializer=initializer, bias_initializer=initializer, recurrent_initializer=initializer, return_sequences=False)(x)
    model = keras.Model(inputs=convolutionInput, outputs=output)
    model.compile(optimizer=optimizer(lr=learningRate), loss="mean_squared_error")

    print("Finished Generating Model")
    return model


#generateNetwork(imageSize=(360, 200), numberOfFilters=[64, 48, 32], filterSizes=[(8, 8), (4, 4), (3, 3)], strideSizes=[(5, 5), (4, 4), (2, 2)], hiddenNodes=[64, 32], outputLength=10)