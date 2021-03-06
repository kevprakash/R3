import tensorflow as tf
from keras.callbacks import TensorBoard
from tensorflow import keras

def initializeRewardNetwork(convolutionNetwork, controllerNetwork, latentSpaceLength, controllerOutputLength, nodesPerLayer, outputLength, dropoutRate=0, optimizer=tf.keras.optimizers.Adam, hiddenActivation='relu', activation='relu', recurrentCell=keras.layers.SimpleRNN, learningRate=0.1):
    initializer = tf.initializers.random_normal

    c0, c1, c2, c3 = convolutionNetwork.input_shape
    convolutionInput = keras.layers.Input(shape=(c0, c1, c2, c3), name="Convolution_Input")
    convolution = keras.layers.TimeDistributed(convolutionNetwork, input_shape=(c0, c1, c2, c3))(convolutionInput)
    convolutionFlattened = recurrentCell(latentSpaceLength, return_sequences=False, activation=hiddenActivation, recurrent_initializer=initializer, kernel_initializer=initializer)(convolution)
    controller = controllerNetwork(convolutionInput)

    sharedModel = keras.Sequential()
    for i in range(0, len(nodesPerLayer)):
        sharedModel.add(keras.layers.Dense(nodesPerLayer[i], activation=hiddenActivation, kernel_initializer=initializer))
        if dropoutRate > 0:
            sharedModel.add(keras.layers.Dropout(rate=dropoutRate))
    sharedModel.add(keras.layers.Dense(outputLength, activation=activation, kernel_initializer=initializer))

    mergeModel = keras.layers.concatenate([convolutionFlattened, controller])
    mergeModel = sharedModel(mergeModel)

    trainerInput = keras.layers.Input(shape=(controllerOutputLength,), dtype='float32', name="Trainer_Input")
    mergeTrainer = keras.layers.concatenate([convolutionFlattened, trainerInput])
    mergeTrainer = sharedModel(mergeTrainer)

    model = keras.Model(inputs=convolutionInput, outputs=sharedModel.output)
    trainer = keras.Model(inputs=[convolutionInput, trainerInput], outputs=sharedModel.output)

    model.compile(optimizer=optimizer(lr=learningRate), loss='mean_squared_error')
    trainer.compile(optimizer=optimizer(lr=learningRate), loss='mean_squared_error')
    return model, trainer

