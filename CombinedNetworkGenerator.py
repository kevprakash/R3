import ControllerNetwork as CN
import ConvolutionalAutoencoder as CAE
import RewardNetwork as RN
import tensorflow as tf

def generateAllNetworks(imageSize, numberOfFilters, filterSizes, latentSpaceLength, controllerNodesPerLayer, controllerOutputLength, rewardNodesPerLayer, rewardOutputLength, dropoutRate=0, optimizer='rmsprop', hiddenActivation=tf.nn.relu, rewardActivation=tf.nn.relu):
    encode, decode, cae = CAE.generateNetworks(imageSize, numberOfFilters=numberOfFilters, filterSizes=filterSizes, latentSpaceLength=latentSpaceLength, hiddenActivation=hiddenActivation, dropoutRate=dropoutRate, learningRate=0.001)
    controller = CN.initializeControllerNetwork(inputModel=encode, latentSpaceLength=latentSpaceLength, nodesPerLayer=controllerNodesPerLayer, outputLength=controllerOutputLength, hiddenActivation=hiddenActivation, dropoutRate=dropoutRate, optimizer=optimizer)
    reward, trainer = RN.initializeRewardNetwork(encode, controller, latentSpaceLength=latentSpaceLength, controllerOutputLength=controllerOutputLength, nodesPerLayer=rewardNodesPerLayer, outputLength=rewardOutputLength, dropoutRate=dropoutRate, optimizer=optimizer, hiddenActivation=hiddenActivation, activation=rewardActivation)
    return encode, decode, cae, controller, reward, trainer
