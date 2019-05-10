import ControllerNetwork as CN
import ConvolutionalAutoencoder as CAE
import RewardNetwork as RN
import tensorflow as tf
from tensorflow import keras


def generateAllNetworks(imageSize, numberOfFilters, filterSizes, strideSizes, latentSpaceLength, controllerNodesPerLayer, controllerOutputLength, rewardNodesPerLayer, rewardOutputLength, dropoutRate=0, optimizer=tf.keras.optimizers.SGD, hiddenActivation=tf.nn.relu, rewardHiddenActivation=tf.nn.tanh, rewardActivation=None, recurrentCell=keras.layers.LSTM, learningRate=0.01):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    encode, decode, cae = CAE.generateNetworks(imageSize, numberOfFilters=numberOfFilters, filterSizes=filterSizes, strideSizes=strideSizes, latentSpaceLength=latentSpaceLength, hiddenActivation=hiddenActivation, dropoutRate=dropoutRate, learningRate=learningRate)
    controller = CN.initializeControllerNetwork(inputModel=encode,  nodesPerLayer=controllerNodesPerLayer, outputLength=controllerOutputLength, hiddenActivation=hiddenActivation, dropoutRate=dropoutRate, optimizer=optimizer, recurrentCell=recurrentCell, learningRate=learningRate)
    reward, trainer = RN.initializeRewardNetwork(encode, controller, latentSpaceLength=latentSpaceLength, controllerOutputLength=controllerOutputLength, nodesPerLayer=rewardNodesPerLayer, outputLength=rewardOutputLength, dropoutRate=dropoutRate, optimizer=optimizer, hiddenActivation=rewardHiddenActivation, activation=rewardActivation, recurrentCell=recurrentCell, learningRate=learningRate)
    return encode, decode, cae, controller, reward, trainer
