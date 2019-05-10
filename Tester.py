import CombinedNetworkGenerator as CNG
import KeyboardOutput as KeyOut
import Trainer
import R3Utilities as R3U
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import ctypes
import math


def testRewardFunction(rawInputs, index):
    raw = rawInputs[index]
    prevRaw = rawInputs[index - 1]
    if index == 0:
        return 0
    if raw[1] <= 0:
        return -100
    reward = 0
    loc = math.sqrt((raw[3] ** 2) + (raw[4] ** 2) * 0.25)
    prevLoc = math.sqrt((prevRaw[3] ** 2) + (prevRaw[4] ** 2) * 0.25)

    reward = reward + min((math.fabs(loc - prevLoc) - 1.5) * 3, 5)

    if prevRaw[1] > 0:
        reward = reward + ((raw[1] + raw[2]) - (prevRaw[1] + prevRaw[2])) * 20
    reward = reward + (raw[0] - prevRaw[0]) * 50

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


def testGameplay(imageSize, outputs, rewardMemoryAddresses, addressTypes, processID, resetScript, iterations=1, processingInterval=0.25, processingIterations=200, maxProcessingIterations = 1000, maxSequenceLength=8, verbose=False):
    inputShape = (imageSize[1], imageSize[0], 3)
    vision, decode, cae, controller, reward, trainer = CNG.generateAllNetworks(imageSize=inputShape, numberOfFilters=[64, 48, 32], filterSizes=[(8, 8), (4, 4), (3, 3)], strideSizes=[(5, 5), (4, 4), (2, 2)], latentSpaceLength=128, controllerNodesPerLayer=[32, 16], controllerOutputLength=len(outputs), rewardNodesPerLayer=[64, 32], rewardOutputLength=1)
    possibleOutputs = np.zeros(shape=(len(outputs), len(outputs)))
    for i in range(len(outputs)):
        possibleOutputs = np.append(possibleOutputs, [R3U.createNHotArray(len(outputs), [i])], axis=0)
        possibleOutputs = np.delete(possibleOutputs, 0, axis=0)

    print(possibleOutputs)

    for count in range(iterations):
        print("Loop", count + 1)
        startTime = time.time()

        i = 0
        previousOutput = -1
        lastSave = time.time()

        length = int(min(processingIterations * max(math.log(max(count, 1), 3), 1), maxProcessingIterations))

        screenCaptures = np.zeros(shape=(1, length, imageSize[1], imageSize[0], 3))
        screenCapturesLimited = np.zeros(shape=(1, maxSequenceLength, imageSize[1], imageSize[0], 3))
        performedOutputs = np.zeros(shape=(length, len(outputs)))
        rewards = None

        while i < length:
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

                #output = controller.predict(screenCapturesLimited)

                output = R3U.createNHotArray(len(outputs), [random.randint(0, len(outputs))])
                outputReward = 0
                for o in possibleOutputs:
                    temp = trainer.predict_on_batch([screenCapturesLimited, np.array([o])])
                    temp = temp[0][0]
                    if (temp > outputReward) or (temp == outputReward and random.uniform(0, 1) < 0.5):
                        output = o
                        outputReward = temp

                outputHex, isMouse = R3U.convertOutputToHex(output, outputs)
                KeyOut.performOutput(outputHex, isMouse, previousOutput)

                performedOutputs = np.append(performedOutputs, [output], axis=0)
                performedOutputs = np.delete(performedOutputs, 0, axis=0)

                if not isMouse:
                    previousOutput = outputHex

                temp = []
                for addr in range(len(rewardMemoryAddresses)):
                    temp.append(R3U.readMemoryAddress(processID, rewardMemoryAddresses[addr], addressTypes[addr]))

                if rewards is None:
                    rewards = np.array([temp])
                else:
                    rewards = np.append(rewards, [temp], axis=0)
                actionReward = testRewardFunction(rewards, i)
                i = i + 1
                R3U.printLoadBar(i/length, 25, endString="%s%s%s%s%s" % (R3U.convertOutputToChar(output, outputs), " => ", str(temp), " => ", actionReward))
        if count == iterations - 1:
            verbose = True
        KeyOut.ReleaseKey(previousOutput)

        duration = time.time() - startTime

        resetScript()

        #print(R3U.generateQArray(rewards, testRewardFunction))
        print("Duration:", int(duration/60), "min", int(duration%60), "s")
        #Trainer.trainNetworks(cae, vision, controller, reward, trainer, screenCaptures, rewards, len(outputs), testRewardFunction, speedUpRate=1, epochs=1, verbose=verbose, subsequenceLength=maxSequenceLength)
        Trainer.trainTrainerNetwork(trainer, screenCaptures, maxSequenceLength, performedOutputs, rewards, testRewardFunction, speedUpRate=1, epochs=1, verbose=True)

def runningNetworksTest():
    vision, decode, cae, controller, reward, trainer = CNG.generateAllNetworks(imageSize=(200, 360, 3), numberOfFilters=[32, 32, 32], filterSizes=[(8, 8), (4, 4), (3, 3)], strideSizes=[(5, 5), (4, 4), (2, 2)], latentSpaceLength=128, controllerNodesPerLayer=[32, 16], controllerOutputLength=10, rewardNodesPerLayer=[32, 16], rewardOutputLength=1)

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

    Trainer.trainNetworks(cae, vision, controller, reward, trainer, screenCaptures, rewards, 10, testRewardFunction, speedUpRate=1, epochs=1, verbose=True, subsequenceLength=20)

keystrokes = list("wasdqe")
keystrokes.extend(["lmouse", "rmouse", "lshift", "space"])
memoryAddresses = [0x92B80118, 0x92B800E4, 0x92B800F0, 0x16D4A7A4, 0x16D4A7A8] #Geo, Health, Shield, xCoord, yCoord in Hollow Knight
types = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float]
testGameplay((360, 200), keystrokes, memoryAddresses, types, processID=0x3894, resetScript=testResetScript, iterations=100000, processingInterval=0.0, processingIterations=50, maxProcessingIterations=200, maxSequenceLength=4, verbose=True)

#runningNetworksTest()

