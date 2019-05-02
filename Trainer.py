import tensorflow as tf
import numpy as np
import R3Utilities as R3U
import ConvolutionalAutoencoder as CAE
import random

def convertReward(rawReward, conversionFunction):
    return conversionFunction(rawReward)

def trainRewardNetwork(rewardNetwork, screenCaptures, rewards, subsequenceLength=5, epochs=5, verbose=False):
    print("Training Reward Network")
    i = 0
    x = None
    y = None
    while i * subsequenceLength < len(rewards):
        end = min((i+1) * subsequenceLength, len(rewards))
        subsequence = np.array([[screenCaptures[0][i * subsequenceLength]]])
        for j in range(i * subsequenceLength + 1, (i + 1) * subsequenceLength):
            if j < end:
                subsequence = np.append(subsequence, [[screenCaptures[0][j]]], axis=1)
            else:
                shape = screenCaptures[0][end - 1].shape
                subsequence = np.append(subsequence, np.zeros(shape=(1, 1, shape[0], shape[1], shape[2])), axis=1)
        if x is None:
            x = np.array(subsequence)
        else:
            x = np.append(x, subsequence, axis=0)
        if y is None:
            y = np.array(rewards[end - 1])
        else:
            y = np.append(y, rewards[end - 1], axis=0)
        i = i + 1

    rewardNetwork.fit(x, y, batch_size=1, epochs=epochs, verbose=verbose, shuffle=False)

def trainControllerNetwork(controllerNetwork, screenCaptures, correctActions, subsequenceLength=5, epochs=5, verbose=False):
    print("Training Controller Network")
    i = 0
    x = None
    y = None
    while i * subsequenceLength < len(correctActions):
        end = min((i + 1) * subsequenceLength, len(correctActions))
        subsequence = np.array([[screenCaptures[0][i * subsequenceLength]]])
        for j in range(i * subsequenceLength + 1, (i + 1) * subsequenceLength + 1):
            if j < end:
                subsequence = np.append(subsequence, [[screenCaptures[0][j]]], axis=1)
            else:
                shape = screenCaptures[0][end - 1].shape
                subsequence = np.append(subsequence, np.zeros(shape=(1, 1, shape[0], shape[1], shape[2])), axis=1)
        if x is None:
            x = np.array(subsequence)
        else:
            x = np.append(x, subsequence, axis=0)
        if y is None:
            y = np.array([correctActions[end - 1]])
        else:
            y = np.append(y, [correctActions[end - 1]], axis=0)
        i = i + 1


    controllerNetwork.fit(x, y, batch_size=1, epochs=epochs, verbose=verbose, shuffle=False)

def predictCorrectAction(trainerNetwork, screenCaptures, numberOfPossibleActions, rewardFunction, subsequenceLength=20):
    print("Predicting Correct Course of Action")
    _, x, y, z = np.shape(screenCaptures[0])
    states = np.zeros(shape=(1, subsequenceLength, x, y, z))
    actions = []
    for i in range(0, len(screenCaptures[0]), int(subsequenceLength/5)):
        for j in range(int(i - subsequenceLength/5), i):
            states = np.append(states, [[screenCaptures[0][j + 1]]], axis=1)
            states = np.delete(states, 0, axis=1)
        maxReward = 0
        maxRewardIndex = 0
        for j in range(numberOfPossibleActions):
            reward = trainerNetwork.predict_on_batch([states, np.array([R3U.createNHotArray(numberOfPossibleActions, [j])])])
            reward = convertReward(reward[0], rewardFunction)
            if j == 0 or reward >= maxReward:
                if reward == maxReward:
                    if random.uniform(0, 1) > (1.0 / numberOfPossibleActions):
                        maxReward = reward
                        maxRewardIndex = j
                else:
                    maxReward = reward
                    maxRewardIndex = j
        actions.append(maxRewardIndex)
        R3U.printLoadBar((i + 1)/len(screenCaptures[0]), length=25)

    return actions

def trainNetworks(convolutionalAutoencoder, controllerNetwork, rewardNetwork, trainerNetwork, screenCaptures, rewards, possibleActions, rewardFunction, verbose=False, epochs=5, subsequenceLength=12):
    print("Beginning Training Sequence")
    CAE.trainCNAE(convolutionalAutoencoder, screenCaptures[0], verbose=verbose, iterations=epochs)
    convWeights = convolutionalAutoencoder.layers
    contWeights = controllerNetwork.layers
    '''for w in convWeights:
        w.trainable = False
    for w in contWeights:
        w.trainable = False'''
    trainRewardNetwork(rewardNetwork, screenCaptures, rewards, subsequenceLength=subsequenceLength, epochs=epochs, verbose=verbose)
    correctActions = predictCorrectAction(trainerNetwork, screenCaptures, possibleActions, rewardFunction, subsequenceLength=subsequenceLength)
    #for w in contWeights:
    #    w.trainable = True
    trainControllerNetwork(controllerNetwork, screenCaptures, correctActions, subsequenceLength=subsequenceLength, epochs=epochs, verbose=verbose)
    #for w in convWeights:
    #    w.trainable = True
    print("Finished Training Sequence")
