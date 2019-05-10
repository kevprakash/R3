import numpy as np
import R3Utilities as R3U
import ConvolutionalAutoencoder as CAE
import random
import time


def generateSubsequences(screenCaptures, subsequenceLength, speedUpRate=2):
    x = None
    _, _, w, h, d = np.shape(screenCaptures)
    subsequence = np.zeros(shape=(1, subsequenceLength, w, h, d))

    print("Generating Training Data Subsequences")
    for i in range(len(screenCaptures[0])):
        subsequence = np.append(subsequence, [[screenCaptures[0][i]]], axis=1)
        subsequence = np.delete(subsequence, 0, axis=1)

        if i % speedUpRate == 0:
            if x is None:
                x = np.array(subsequence)
            else:
                x = np.append(x, subsequence, axis=0)

        R3U.printLoadBar((i + 1) / len(screenCaptures[0]), length=25)

    return x


def trainRewardNetwork(rewardNetwork, states, rewards, rewardFunction, speedUpRate=2, epochs=5, verbose=False):

    rewardForSequence = R3U.generateQArray(rewards, rewardFunction, speedUpRate=speedUpRate)

    print("Training Reward Network")
    rewardNetwork.fit(states, rewardForSequence, batch_size=1, epochs=epochs, verbose=verbose, shuffle=False)


def trainControllerNetwork(convolutionNetwork, controllerNetwork, states, correctActions, epochs=5, verbose=False):
    y = None
    print("Creating Training Data for Controller Network")
    actionIndex = 0
    for i in range(len(states)):
        if y is None:
            y = np.array([correctActions[actionIndex]])
        else:
            y = np.append(y, [correctActions[actionIndex]], axis=0)
        actionIndex = actionIndex + 1
        R3U.printLoadBar((i + 1)/len(states), length=25)

    print("Training Controller Network")

    for layer in controllerNetwork.layers:
        layer.trainable = True

    for layer in convolutionNetwork.layers:
        layer.trainable = True

    controllerNetwork.fit(states, y, batch_size=1, epochs=epochs*5, verbose=verbose, shuffle=False)

    for layer in controllerNetwork.layers:
        layer.trainable = False

    for layer in convolutionNetwork.layers:
        layer.trainable = False

def predictCorrectAction(trainerNetwork, states, numberOfPossibleActions):
    print("Predicting Correct Course of Action")
    actions = []
    for i in range(len(states)):

        maxReward = -1
        maxRewardIndex = 0
        for j in range(numberOfPossibleActions):
            testAction = R3U.createNHotArray(numberOfPossibleActions, [j])
            testReward = trainerNetwork.predict_on_batch([[states[i]], np.array([testAction])])
            if testReward[0][0] > maxReward:
                maxRewardIndex = j
                maxReward = testReward[0][0]
        if maxReward < 0:
            maxRewardIndex = random.randint(0, numberOfPossibleActions - 1)

        R3U.printLoadBar((i + 1)/len(states), length=25)

        actions.append(maxRewardIndex)

    return actions


def trainTrainerNetwork(trainerNetwork, screenCaptures, subsequenceLength, actions, rewards, rewardFunction, speedUpRate=2, epochs=5, verbose=False):
    states = generateSubsequences(screenCaptures, subsequenceLength, speedUpRate=speedUpRate)
    rewardForSequence = R3U.generateQArray(rewards, rewardFunction, speedUpRate=speedUpRate)

    print("Training Reward Network")
    trainerNetwork.fit([states, actions], rewardForSequence, batch_size=1, epochs=epochs, verbose=verbose, shuffle=False)


def trainNetworks(convolutionalAutoencoder, vision, controllerNetwork, rewardNetwork, trainerNetwork, screenCaptures, rewards, possibleActions, rewardFunction, speedUpRate=2, verbose=False, epochs=5, subsequenceLength=12):
    print("Beginning Training Sequence")
    startTime = time.time()
    #wwCAE.trainCNAE(convolutionalAutoencoder, screenCaptures[0], verbose=verbose, iterations=epochs)
    states = generateSubsequences(screenCaptures, subsequenceLength, speedUpRate=speedUpRate)
    trainRewardNetwork(rewardNetwork, states, rewards, speedUpRate=speedUpRate, rewardFunction=rewardFunction, epochs=epochs, verbose=verbose)
    correctActions = predictCorrectAction(trainerNetwork, states, possibleActions)
    trainControllerNetwork(vision, controllerNetwork, states, correctActions, epochs=epochs, verbose=verbose)
    duration = time.time() - startTime
    print("Finished Training Sequence:", int(duration / 60), "min", int(duration % 60), "s")
