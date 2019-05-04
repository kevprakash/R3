import CombinedNetworkGenerator as CNG
import KeyboardOutput as KeyOut
import Trainer
import R3Utilities as R3U
import numpy as np
import time

def testRewardFunction(rawInputs, index):
    raw = rawInputs[index]
    if index == 0:
        return 0
    if raw[1] <= 0:
        return -100
    reward = 0
    prevRaw = rawInputs[index - 1]
    if raw[3] == prevRaw[3]:
        reward = reward - 5
    reward = reward + ((raw[1] + raw[2]) - (prevRaw[1] + prevRaw[2])) * 20
    reward = reward + (raw[0] - prevRaw[0]) * 50

    return reward


def testResetScript():
    time.sleep(1)

    KeyOut.PressKey(R3U.convertCharToHex('w')[0])
    KeyOut.PressKey(R3U.convertCharToHex('f')[0])
    time.sleep(5)
    KeyOut.ReleaseKey(R3U.convertCharToHex('w')[0])
    KeyOut.ReleaseKey(R3U.convertCharToHex('f')[0])

    time.sleep(5)
    KeyOut.PressKey(R3U.convertCharToHex('w')[0])
    KeyOut.ReleaseKey(R3U.convertCharToHex('w')[0])
    time.sleep(0.5)
    KeyOut.PressKey(R3U.convertCharToHex('w')[0])
    KeyOut.ReleaseKey(R3U.convertCharToHex('w')[0])

    time.sleep(1)
    KeyOut.click(R3U.convertCharToHex("lmouse")[0])


def testGameplay(imageSize, outputs, rewardMemoryAddresses, processID, resetScript, iterations=1, processingInterval=0.25, processingIterations=60, maxSequenceLength=8):
    inputShape = (imageSize[0], imageSize[1], 3)
    vision, decode, cae, controller, reward, trainer = CNG.generateAllNetworks(imageSize=inputShape, numberOfFilters=[32, 32, 32], filterSizes=[(5, 5), (4, 4), (2, 2)], latentSpaceLength=128, controllerNodesPerLayer=[32], controllerOutputLength=len(outputs), rewardNodesPerLayer=[32], rewardOutputLength=1)
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
                if not isMouse:
                    previousOutput = outputHex

                temp = []
                for addr in range(len(rewardMemoryAddresses)):
                    temp.append(R3U.readMemoryAddress(processID, rewardMemoryAddresses[addr]))

                if rewards is None:
                    rewards = np.array([temp])
                else:
                    rewards = np.append(rewards, [temp], axis=0)
                lastSave = time.time()
                actionReward = testRewardFunction(rewards, i)
                i = i + 1
                R3U.printLoadBar(i/processingIterations, 25, endString="%s%s%s%s%s" % (R3U.convertOutputToChar(output[-1], outputs), " => ", str(temp), "=>", actionReward))
        verbose = False
        if count == iterations - 1:
            verbose = True
        KeyOut.ReleaseKey(previousOutput)

        resetScript()

        Trainer.trainNetworks(cae, controller, reward, trainer, screenCaptures, rewards, len(outputs), testRewardFunction, epochs=5, verbose=verbose, subsequenceLength=maxSequenceLength)


keystrokes = list("wasdqfe")
keystrokes.extend(["lmouse", "rmouse", "lshift", "space"])
rewardMemoryAddresses = [0x93BD6118, 0x93BD60E4, 0x93BD60F0, 0x14186188] #Geo, Health, Shield, xCoord in Hollow Knight
testGameplay((360, 200), keystrokes, rewardMemoryAddresses, processID=0x3270, resetScript=testResetScript, iterations=100000, processingInterval=0.15, processingIterations=200, maxSequenceLength=20)