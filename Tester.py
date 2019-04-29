import CombinedNetworkGenerator as CNG
import KeyboardOutput as KeyOut
import Trainer
import random
import R3Utilities as R3U
import numpy as np
import time

def testRewardFunction(rawInput):
    return rawInput[0] - rawInput[1]

def testGameplay(imageSize, outputs, iterations=1, saveInterval=0.25, timeOutDuration=60, maxSequenceLength=8):
    inputShape = (imageSize[0], imageSize[1], 3)
    vision, decode, cae, controller, reward, trainer = CNG.generateAllNetworks(imageSize=inputShape, numberOfFilters=[16, 16, 16], filterSizes=[(2, 2), (2, 2), (2, 2)], latentSpaceLength=128, controllerNodesPerLayer=[16, 8], controllerOutputLength=len(outputs), rewardNodesPerLayer=[16, 8], rewardOutputLength=2)
    for count in range(iterations):
        print("Loop", count + 1)
        i = 0.0
        previousOutput = -1
        screenCaptures = None
        screenCapturesLimited = None
        rewards = None
        startTime = time.time()
        lastSave = time.time()
        barUpdateTime = time.time()
        while time.time() - startTime < timeOutDuration:
            if time.time() - lastSave >= saveInterval:
                screen = R3U.takeScreenShot(0, 0, 1920, 1080, imageSize)
                screen = np.reshape(screen, newshape=(1, screen.shape[1], screen.shape[0], screen.shape[2]))
                if screenCaptures is None:
                    screenCaptures = np.array([screen])
                else:
                    screenCaptures = np.append(screenCaptures, [screen], axis=1)
                if screenCapturesLimited is None:
                    screenCapturesLimited = np.array([screen])
                else:
                    screenCapturesLimited = np.append(screenCapturesLimited, [screen], axis=1)
                    if len(screenCapturesLimited) == maxSequenceLength:
                        screenCapturesLimited = np.delete(screenCapturesLimited, 0, axis=1)
                output = controller.predict(screenCapturesLimited)
                outputHex = R3U.convertOutputToHex(output[-1], outputs)
                #KeyOut.performOutput(outputHex, previousOutput)
                previousOutput = outputHex
                if rewards is None:
                    rewards = np.array([[[random.randint(0, 10), random.randint(0, 10)]]])
                else:
                    rewards = np.append(rewards, [[[random.randint(0, 10), random.randint(0, 10)]]], axis=0)
                lastSave = time.time()
            if time.time() - barUpdateTime > 1:
                i = i + time.time() - barUpdateTime
                R3U.printLoadBar(i/timeOutDuration, 25)
                barUpdateTime = time.time()
        verbose = False
        if count == iterations - 1:
            verbose = True
        Trainer.trainNetworks(cae, controller, reward, trainer, screenCaptures, rewards, len(outputs), testRewardFunction, verbose=verbose)

keystrokes = list('wasd')
testGameplay((360, 200), keystrokes, iterations=5, saveInterval=0.25, timeOutDuration=9, maxSequenceLength=12)