import CombinedNetworkGenerator as CNG
import KeyboardOutput as KeyOut
import Trainer
import R3Utilities as R3U
import numpy as np
import time

def testRewardFunction(rawInputs, startTime, stopTime):
    reward = 0
    prevGeo = 0
    prevCombinedHealth = 17
    prevX = -1
    for i in range(startTime, stopTime):
        if rawInputs[i][1] > 0:
            reward = reward + (rawInputs[i][0] - prevGeo) * (prevCombinedHealth - (rawInputs[i][1] + rawInputs[i][2]))
            prevGeo = rawInputs[i][0]
            prevCombinedHealth = rawInputs[i][1] + rawInputs[i][2]

            if prevX > 0:
                if prevX == rawInputs[i][3]:
                    if reward < 0:
                        reward = reward * 1.1
                    elif reward < 2:
                        reward = reward - 1
                    else:
                        reward = reward/2
                else:
                    reward = reward + 1
            prevX = rawInputs[i][3]

        else:
            reward = 0

    return reward

def testGameplay(imageSize, outputs, rewardMemoryAddresses, processID, iterations=1, processingInterval=0.25, processingIterations=60, maxSequenceLength=8):
    inputShape = (imageSize[0], imageSize[1], 3)
    vision, decode, cae, controller, reward, trainer = CNG.generateAllNetworks(imageSize=inputShape, numberOfFilters=[16, 16, 16], filterSizes=[(5, 5), (4, 4), (2, 2)], latentSpaceLength=128, controllerNodesPerLayer=[32], controllerOutputLength=len(outputs), rewardNodesPerLayer=[32], rewardOutputLength=1)
    for count in range(iterations):
        print("Loop", count + 1)

        i = 0
        previousOutput = -1
        lastSave = time.time()

        screenCaptures = np.zeros(shape=(1, processingIterations, imageSize[0], imageSize[1], 3))
        screenCapturesLimited = np.zeros(shape=(1, maxSequenceLength, imageSize[0], imageSize[1], 3))
        rewards = None
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

                if rewards is None:
                    rewards = np.array([temp])
                else:
                    rewards = np.append(rewards, [temp], axis=0)
                lastSave = time.time()
                actionReward = testRewardFunction(rewards, max(0, i - maxSequenceLength), i + 1)
                i = i + 1
                R3U.printLoadBar(i/processingIterations, 25, endString="%s%s%s%s%s" % (R3U.convertOutputToChar(output[-1], outputs), " => ", str(temp), "=>", actionReward))
        verbose = False
        if count == iterations - 1:
            verbose = True
        KeyOut.ReleaseKey(previousOutput)

        KeyOut.PressKey(R3U.convertCharToHex('w')[0])
        KeyOut.PressKey(R3U.convertCharToHex('f')[0])
        time.sleep(5)
        KeyOut.ReleaseKey(R3U.convertCharToHex('w')[0])
        KeyOut.ReleaseKey(R3U.convertCharToHex('f')[0])

        Trainer.trainNetworks(cae, controller, reward, trainer, screenCaptures, rewards, len(outputs), testRewardFunction, epochs=5, verbose=verbose, subsequenceLength=maxSequenceLength)

keystrokes = list("wasdqfe")
keystrokes.extend(["lmouse", "rmouse", "lshift", "space"])
rewardMemoryAddresses = [0x93CAD118, 0x93CAD0E4, 0x0423CBD0, 0x04D9928C] #Geo, Health, Shield, xCoord in Hollow Knight
testGameplay((360, 200), keystrokes, rewardMemoryAddresses, processID=0x1928, iterations=100000, processingInterval=0.15, processingIterations=20, maxSequenceLength=20)