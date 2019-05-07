import CombinedNetworkGenerator as CNG
import KeyboardOutput as KeyOut
import Trainer
import R3Utilities as R3U
import numpy as np
import time
import matplotlib.pyplot as plt
import random

def testRewardFunction(rawInputs, index):
    raw = rawInputs[index]
    if index == 0:
        return 0
    if raw[1] <= 0:
        return -100 - (rawInputs[index - 1][0] * 10)
    reward = 0
    prevRaw = rawInputs[index - 1]
    for i in range(max(0, index - 10), index):
        if raw[3] == rawInputs[i][3]:
            reward = reward - (2 / (index - i))
    reward = reward + 1
    reward = reward + ((raw[1] + raw[2]) - (prevRaw[1] + prevRaw[2])) * 10
    reward = reward + (raw[0] - prevRaw[0]) * 20

    return reward


def testResetScript():

    KeyOut.click(R3U.convertCharToHex('lmouse')[0])
    time.sleep(0.25)

    KeyOut.PressKey(R3U.convertCharToHex('f')[0])
    KeyOut.PressKey(R3U.convertCharToHex('w')[0])
    time.sleep(5)
    KeyOut.ReleaseKey(R3U.convertCharToHex('w')[0])
    KeyOut.ReleaseKey(R3U.convertCharToHex('f')[0])

    time.sleep(5)
    KeyOut.PressKey(R3U.convertCharToHex('w')[0])
    KeyOut.ReleaseKey(R3U.convertCharToHex('w')[0])
    time.sleep(0.5)
    KeyOut.PressKey(R3U.convertCharToHex('w')[0])
    KeyOut.ReleaseKey(R3U.convertCharToHex('w')[0])


def testGameplay(imageSize, outputs, rewardMemoryAddresses, processID, resetScript, iterations=1, processingInterval=0.25, processingIterations=60, maxSequenceLength=8):
    inputShape = (imageSize[1], imageSize[0], 3)
    vision, decode, cae, controller, reward, trainer = CNG.generateAllNetworks(imageSize=inputShape, numberOfFilters=[32, 32, 32], filterSizes=[(5, 5), (4, 4), (2, 2)], latentSpaceLength=128, controllerNodesPerLayer=[32, 16], controllerOutputLength=len(outputs), rewardNodesPerLayer=[32, 16], rewardOutputLength=1)
    for count in range(iterations):
        print("Loop", count + 1)
        startTime = time.time()

        i = 0
        previousOutput = -1
        lastSave = time.time()

        screenCaptures = np.zeros(shape=(1, processingIterations, imageSize[1], imageSize[0], 3))
        screenCapturesLimited = np.zeros(shape=(1, maxSequenceLength, imageSize[1], imageSize[0], 3))
        rewards = None
        while i < processingIterations:
            if time.time() - lastSave >= processingInterval:
                lastSave = time.time()

                screen = R3U.takeScreenShot(0, 0, 1920, 1080, imageSize)
                screen = np.reshape(screen, newshape=(1, screen.shape[0], screen.shape[1], screen.shape[2]))

                #plt.imshow(screen[0], cmap=plt.cm.hsv)
                #plt.show()

                screenCaptures = np.append(screenCaptures, [screen], axis=1)
                screenCaptures = np.delete(screenCaptures, 0, axis=1)

                screenCapturesLimited = np.append(screenCapturesLimited, [screen], axis=1)
                screenCapturesLimited = np.delete(screenCapturesLimited, 0, axis=1)

                output = controller.predict(screenCapturesLimited)

                outputHex, isMouse = R3U.convertOutputToHex(output[0], outputs)
                KeyOut.performOutput(outputHex, isMouse, previousOutput)
                if not isMouse:
                    previousOutput = outputHex

                temp = []
                for addr in range(len(rewardMemoryAddresses)):
                    temp.append(R3U.readMemoryAddress(processID, rewardMemoryAddresses[addr]))

                if rewards is None:
                    rewards = np.array([temp])
                else:
                    rewards = np.append(rewards, [temp], axis=0)
                actionReward = testRewardFunction(rewards, i)
                i = i + 1
                R3U.printLoadBar(i/processingIterations, 25, endString="%s%s%s%s%s" % (R3U.convertOutputToChar(output[-1], outputs), " => ", str(temp), "=>", actionReward))
        verbose = False
        if count == iterations - 1:
            verbose = True
        KeyOut.ReleaseKey(previousOutput)

        duration = time.time() - startTime

        resetScript()

        #print(R3U.generateQArray(rewards, testRewardFunction))
        print("Duration:", int(duration/60), "min", int(duration%60), "s")
        Trainer.trainNetworks(cae, controller, reward, trainer, screenCaptures, rewards, len(outputs), testRewardFunction, speedUpRate=4, epochs=5, verbose=verbose, subsequenceLength=maxSequenceLength)

def runningNetworksTest():
    vision, decode, cae, controller, reward, trainer = CNG.generateAllNetworks(imageSize=(200, 360, 3), numberOfFilters=[32, 32, 32], filterSizes=[(5, 5), (4, 4), (2, 2)], latentSpaceLength=32, controllerNodesPerLayer=[32, 16], controllerOutputLength=10, rewardNodesPerLayer=[32, 16], rewardOutputLength=1)

    screenCaptures = np.zeros(shape=(1, 20, 200, 360, 3))
    rewards = None

    for i in range(20):
        screen = R3U.takeScreenShot(0, 0, 1920, 1080, (360, 200))
        screen = np.reshape(screen, newshape=(1, screen.shape[0], screen.shape[1], screen.shape[2]))

        if i == 0:
            plt.imshow(screen[0], cmap=plt.cm.hsv)
            plt.show()

        screenCaptures = np.append(screenCaptures, [screen], axis=1)
        screenCaptures = np.delete(screenCaptures, 0, axis=1)

        temp = []
        for addr in range(4):
            temp.append(random.uniform(0, 10))

        if rewards is None:
            rewards = np.array([temp])
        else:
            rewards = np.append(rewards, [temp], axis=0)

    encoded = vision.predict(screenCaptures[0])
    print(np.shape(screenCaptures[0]))
    print(np.shape(encoded))

    output = controller.predict(screenCaptures)
    print(np.shape(output))

    q = R3U.generateQArray(rewards, testRewardFunction)

    predictedRewards = reward.predict(screenCaptures)
    print(np.shape(predictedRewards))

    testInput = R3U.createNHotArray(10, [int(random.random() * 10)])
    predictedInputRewards = trainer.predict_on_batch([screenCaptures, output])
    print(np.shape(predictedInputRewards), predictedRewards[0][0], predictedInputRewards[0][0])
    print(np.shape(trainer.predict_on_batch([screenCaptures, np.array([testInput])])))


keystrokes = list("wasdqe")
keystrokes.extend(["lmouse", "rmouse", "space"])
rewardMemoryAddresses = [0x93BA0118, 0x93BA00E4, 0x93BA00F0, 0x138A3188] #Geo, Health, Shield, xCoord in Hollow Knight
testGameplay((360, 200), keystrokes, rewardMemoryAddresses, processID=0x3574, resetScript=testResetScript, iterations=100000, processingInterval=0.1, processingIterations=200, maxSequenceLength=4)\

#runningNetworksTest()

