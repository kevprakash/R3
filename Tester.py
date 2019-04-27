import CombinedNetworkGenerator as CNG
import KeyboardOutput as KeyOut
import R3Utilities as R3U
import numpy as np
import time

def testGameplay(imageSize, outputs, saveInterval=0):
    inputShape = (imageSize[0], imageSize[1], 3)
    vision, decode, cae, controller, reward, trainer = CNG.generateAllNetworks(imageSize=inputShape, numberOfFilters=[32, 32, 32], filterSizes=[(2, 2), (2, 2), (2, 2)], latentSpaceLength=128, controllerNodesPerLayer=[64, 32], controllerOutputLength=len(outputs), rewardNodesPerLayer=[16, 16], rewardOutputLength=2)
    i = 0
    previousOutput = -1
    screenCaptures = []
    actions = []
    lastSave = time.time()
    while True:
        screen = R3U.takeScreenShot(0, 0, 1920, 1080, imageSize)
        screen = np.reshape(screen, newshape=(1, screen.shape[1], screen.shape[0], screen.shape[2]))
        output = controller.predict(screen)
        outputHex = R3U.convertOutputToHex(output, outputs)
        KeyOut.performOutput(outputHex, previousOutput)
        previousOutput = outputHex
        if time.time() - lastSave >= saveInterval:
            screenCaptures.append(screen)
            actions.append(outputHex)
            lastSave = time.time()
        R3U.printLoadBar(((i % 100) + 1)/100, 20)
        i = i + 1

keystrokes = list('wasd')
testGameplay((360, 200), keystrokes)