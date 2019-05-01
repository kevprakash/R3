import CombinedNetworkGenerator as CNG
import KeyboardOutput as KeyOut
import Trainer
import random
import R3Utilities as R3U
import numpy as np
import time

def testRewardFunction(rawInputs):
    prod = 1
    for raw in rawInputs:
        prod = prod * raw
    return prod

def testGameplay(imageSize, outputs, rewardMemoryAddresses, processID, iterations=1, processingInterval=0.25, processingIterations=60, maxSequenceLength=8):
    inputShape = (imageSize[0], imageSize[1], 3)
    vision, decode, cae, controller, reward, trainer = CNG.generateAllNetworks(imageSize=inputShape, numberOfFilters=[16, 16, 16], filterSizes=[(2, 2), (2, 2), (2, 2)], latentSpaceLength=128, controllerNodesPerLayer=[16, 8], controllerOutputLength=len(outputs), rewardNodesPerLayer=[16, 8], rewardOutputLength=len(rewardMemoryAddresses))
    for count in range(iterations):
        print("Loop", count + 1)

        i = 0.0
        previousOutput = -1
        lastSave = time.time()

        screenCaptures = np.zeros(shape=(1, processingIterations, imageSize[0], imageSize[1], 3))
        screenCapturesLimited = np.zeros(shape=(1, maxSequenceLength, imageSize[0], imageSize[1], 3))
        rewards = np.zeros(shape=(processingIterations, 1, len(rewardMemoryAddresses)))
        while i < processingIterations:
            if time.time() - lastSave >= processingInterval:
                screen = R3U.takeScreenShot(0, 0, 1920, 1080, imageSize)
                screen = np.reshape(screen, newshape=(1, screen.shape[1], screen.shape[0], screen.shape[2]))

                screenCaptures = np.append(screenCaptures, [screen], axis=1)
                screenCaptures = np.delete(screenCaptures, 0, axis=1)

                screenCapturesLimited = np.append(screenCapturesLimited, [screen], axis=1)
                screenCapturesLimited = np.delete(screenCapturesLimited, 0, axis=1)

                output = controller.predict(screenCapturesLimited)
                outputHex, isMouse = R3U.convertOutputToHex(output[-1], outputs)
                KeyOut.performOutput(outputHex, isMouse, previousOutput)
                previousOutput = outputHex

                temp = []
                for addr in range(len(rewardMemoryAddresses)):
                    temp.append(R3U.readMemoryAddress(processID, rewardMemoryAddresses[addr]))

                rewards = np.append(rewards, [[temp]], axis=0)
                rewards = np.delete(rewards, 0, axis=0)
                lastSave = time.time()
                i = i + 1
                R3U.printLoadBar(i/processingIterations, 25)
        verbose = False
        if count == iterations - 1:
            verbose = True
        Trainer.trainNetworks(cae, controller, reward, trainer, screenCaptures, rewards, len(outputs), testRewardFunction, epochs=5, verbose=verbose, subsequenceLengthTraining=maxSequenceLength)

keystrokes = list("wasdqgfe")
keystrokes.extend(["lmouse", "rmouse", "lshift", "tab"])
keystrokes = ["lmouse", "rmouse"]
rewardMemoryAddresses = [0x93077118, 0x930770E4]
testGameplay((360, 200), keystrokes, rewardMemoryAddresses, processID=0x6A0, iterations=600, processingInterval=0.1, processingIterations=10, maxSequenceLength=12)