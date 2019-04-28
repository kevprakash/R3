import CombinedNetworkGenerator as CNG
import KeyboardOutput as KeyOut
import Trainer
import random
import R3Utilities as R3U
import numpy as np
import time

def testRewardFunction(rawInput):
    return rawInput[0] - rawInput[1]

def testGameplay(imageSize, outputs, iterations=1, saveInterval=0, timeOutDuration=10):
    inputShape = (imageSize[0], imageSize[1], 3)
    vision, decode, cae, controller, reward, trainer = CNG.generateAllNetworks(imageSize=inputShape, numberOfFilters=[32, 32, 32], filterSizes=[(2, 2), (2, 2), (2, 2)], latentSpaceLength=128, controllerNodesPerLayer=[64, 32], controllerOutputLength=len(outputs), rewardNodesPerLayer=[16, 16], rewardOutputLength=2)
    for _ in range(iterations):
        i = 0
        previousOutput = -1
        screenCaptures = None
        rewards = None
        startTime = time.time()
        lastSave = time.time()
        while time.time() - startTime < timeOutDuration:
            if time.time() - lastSave >= saveInterval:
                screen = R3U.takeScreenShot(0, 0, 1920, 1080, imageSize)
                screen = np.reshape(screen, newshape=(1, screen.shape[1], screen.shape[0], screen.shape[2]))
                if screenCaptures is None:
                    screenCaptures = np.array([screen])
                else:
                    screenCaptures = np.append(screenCaptures, [screen], axis=0)
                output = controller.predict([screenCaptures])
                outputHex = R3U.convertOutputToHex(output[-1], outputs)
                #KeyOut.performOutput(outputHex, previousOutput)
                previousOutput = outputHex
                if rewards is None:
                    rewards = np.array([[random.randint(0, 10), random.randint(0, 10)]])
                else:
                    rewards = np.append(rewards, [[random.randint(0, 10), random.randint(0, 10)]], axis=0)
                lastSave = time.time()
            R3U.printLoadBar(((i % 100) + 1)/100, 20)
            i = i + 1
        Trainer.trainRewardNetwork(reward, screenCaptures, rewards, verbose=True)
        Trainer.predictCorrectAction(trainer, screenCaptures, outputs, testRewardFunction)

keystrokes = list('wasd')
testGameplay((360, 200), keystrokes)