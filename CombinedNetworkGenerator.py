import ControllerNetwork as CN
import ConvolutionalAutoencoder as CAE
import RewardNetwork as RN
import tensorflow as tf

def generateAllNetworks(imageSize, numberOfFilters, filterSizes, latentSpaceLength, controllerNodesPerLayer, controllerOutputLength, rewardNodesPerLayer, rewardOutputLength, rewardLossFunction, dropoutRate=0, optimizer='rmsprop', hiddenActivation=tf.nn.relu, activation=tf.nn.sigmoid):
    encode, decode, cae = CAE.generateNetworks(imageSize, numberOfFilters=numberOfFilters, filterSizes=filterSizes, latentSpaceLength=latentSpaceLength, hiddenActivation=hiddenActivation, dropoutRate=dropoutRate)
    controller = CN.initializeControllerNetwork(inputModel=encode, latentSpaceLength=latentSpaceLength, nodesPerLayer=controllerNodesPerLayer, outputLength=controllerOutputLength, hiddenActivation=hiddenActivation, activation=activation, dropoutRate=dropoutRate, optimizer=optimizer)
    reward = RN.initializeRewardNetwork(encode, controller, latentSpaceLength=latentSpaceLength, controllerOutputLength=controllerOutputLength, nodesPerLayer=rewardNodesPerLayer, lossFunction=rewardLossFunction, outputLength=rewardOutputLength, dropoutRate=dropoutRate, optimizer=optimizer, hiddenActivation=hiddenActivation, activation=activation)
    return encode, decode, cae, controller, reward
