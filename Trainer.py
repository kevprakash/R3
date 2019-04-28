import tensorflow as tf
import numpy as np
import R3Utilities as R3U

def convertReward(rawReward, conversionFunction):
    return conversionFunction(rawReward)

def trainRewardNetwork(rewardNetwork, screencaptures, rewards, verbose=False):
    rewardNetwork.fit(screencaptures, rewards, batch_size=1, verbose=verbose, shuffle=False)

def predictCorrectAction(trainerNetwork, screencaptures, possibleActions, rewardFunction):
    states = []
    actions = []
    for i in range(len(screencaptures)):
        states.append(screencaptures[i])
        maxReward = 0
        maxRewardIndex = 0
        for j in range(len(possibleActions)):
            reward = trainerNetwork.predict_on_batch([np.array(states), np.array([R3U.createNHotArray(len(possibleActions), [j])])])
            reward = convertReward(reward[0], rewardFunction)
            if j == 0 or reward > maxReward:
                maxReward = reward
                maxRewardIndex = j
        actions.append(possibleActions[maxRewardIndex])
    return actions