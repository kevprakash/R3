import numpy as np
import R3Utilities as R3U
import ConvolutionalAutoencoder as CAE
import random
import time

def trainRewardNetwork(rewardNetwork, screenCaptures, rewards, rewardFunction, subsequenceLength=5, epochs=5, verbose=False):
    x = None
    _, _, w, h, d = np.shape(screenCaptures)
    subsequence = np.zeros(shape=(1, subsequenceLength, w, h, d))
    print("Generating Traing Data for Reward Network")
    for i in range(len(screenCaptures[0])):
        subsequence = np.append(subsequence, [[screenCaptures[0][i]]], axis=1)
        subsequence = np.delete(subsequence, 0, axis=1)

        if x is None:
            x = np.array(subsequence)
        else:
            x = np.append(x, subsequence, axis=0)

        R3U.printLoadBar((i + 1) / len(screenCaptures[0]), length=25)

    rewardForSequence = R3U.generateQArray(rewards, rewardFunction)

    print("Training Reward Network")
    rewardNetwork.fit(x, rewardForSequence, batch_size=1, epochs=epochs, verbose=verbose, shuffle=False)


def trainControllerNetwork(controllerNetwork, screenCaptures, correctActions, subsequenceLength=5, epochs=5, verbose=False):
    x = None
    y = None
    _, _, w, h, d = np.shape(screenCaptures)
    subsequence = np.zeros(shape=(1, subsequenceLength, w, h, d))
    print("Creating Training Data for Controller Network")
    actionIndex = 0
    for i in range(len(screenCaptures[0])):
        subsequence = np.append(subsequence, [[screenCaptures[0][i]]], axis=1)
        subsequence = np.delete(subsequence, 0, axis=1)

        if x is None:
            x = np.array(subsequence)
        else:
            x = np.append(x, subsequence, axis=0)

        if y is None:
            y = np.array([correctActions[actionIndex]])
        else:
            y = np.append(y, [correctActions[actionIndex]], axis=0)
        actionIndex = actionIndex + 1
        R3U.printLoadBar((i + 1)/len(screenCaptures[0]), length=25)

    print("Training Controller Network")
    controllerNetwork.fit(x, y, batch_size=1, epochs=epochs, verbose=verbose, shuffle=False)


def predictCorrectAction(trainerNetwork, screenCaptures, numberOfPossibleActions, subsequenceLength=20):
    print("Predicting Correct Course of Action")
    _, _, x, y, z = np.shape(screenCaptures)
    states = np.zeros(shape=(1, subsequenceLength, x, y, z))
    actions = []
    for i in range(len(screenCaptures[0])):
        states = np.append(states, [[screenCaptures[0][i]]], axis=1)
        states = np.delete(states, 0, axis=1)

        maxReward = 0
        maxRewardIndex = 0
        for j in range(numberOfPossibleActions):
            testAction = R3U.createNHotArray(numberOfPossibleActions, [j])
            testReward = trainerNetwork.predict_on_batch([states, np.array([testAction])])
            if testReward > maxReward or (maxReward < 0 and random.uniform(0, 1) < 0.5):
                maxRewardIndex = j
                maxReward = testReward

        R3U.printLoadBar((i + 1)/len(screenCaptures[0]), length=25)

        actions.append(maxRewardIndex)

    return actions


def trainNetworks(convolutionalAutoencoder, controllerNetwork, rewardNetwork, trainerNetwork, screenCaptures, rewards, possibleActions, rewardFunction, verbose=False, epochs=5, subsequenceLength=12):
    print("Beginning Training Sequence")
    startTime = time.time()
    CAE.trainCNAE(convolutionalAutoencoder, screenCaptures[0], verbose=verbose, iterations=epochs)
    trainRewardNetwork(rewardNetwork, screenCaptures, rewards, rewardFunction=rewardFunction, subsequenceLength=subsequenceLength, epochs=epochs, verbose=verbose)
    correctActions = predictCorrectAction(trainerNetwork, screenCaptures, possibleActions, subsequenceLength=subsequenceLength)
    trainControllerNetwork(controllerNetwork, screenCaptures, correctActions, subsequenceLength=subsequenceLength, epochs=epochs, verbose=verbose)
    endTime = time.time()
    print("Finished Training Sequence:", (endTime - startTime)/60, "minutes", (endTime - startTime)%60, "seconds")
