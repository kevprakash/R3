import ControllerNetwork as CN
import ConvolutionalAutoencoder as CAE
import RewardNetwork as RN
import tensorflow as tf
from tensorflow import keras


def generateAllNetworks(imageSize, numberOfFilters, filterSizes, latentSpaceLength, controllerNodesPerLayer, controllerOutputLength, rewardNodesPerLayer, rewardOutputLength, dropoutRate=0, optimizer='rmsprop', hiddenActivation=tf.nn.relu, rewardActivation=tf.nn.relu, recurrentCell=keras.layers.SimpleRNN):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    encode, decode, cae = CAE.generateNetworks(imageSize, numberOfFilters=numberOfFilters, filterSizes=filterSizes, latentSpaceLength=latentSpaceLength, hiddenActivation=hiddenActivation, dropoutRate=dropoutRate, learningRate=0.001)
    controller = CN.initializeControllerNetwork(inputModel=encode, latentSpaceLength=latentSpaceLength, nodesPerLayer=controllerNodesPerLayer, outputLength=controllerOutputLength, hiddenActivation=hiddenActivation, dropoutRate=dropoutRate, optimizer=optimizer, recurrentCell=recurrentCell)
    reward, trainer = RN.initializeRewardNetwork(encode, controller, latentSpaceLength=latentSpaceLength, controllerOutputLength=controllerOutputLength, nodesPerLayer=rewardNodesPerLayer, outputLength=rewardOutputLength, dropoutRate=dropoutRate, optimizer=optimizer, hiddenActivation=hiddenActivation, activation=rewardActivation, recurrentCell=recurrentCell)
    return encode, decode, cae, controller, reward, trainer
