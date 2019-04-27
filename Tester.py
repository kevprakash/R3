import CombinedNetworkGenerator as CNG
import KeyboardOutput as KeyOut
import OutputConverter as OC
import R3Utilities as R3U
import numpy as np
import time

def testGameplay(imageSize, outputs):
    inputShape = (imageSize[0], imageSize[1], 3)
    vision, decode, cae, controller, reward = CNG.generateAllNetworks(imageSize=inputShape, numberOfFilters=[32, 32, 32], filterSizes=[(2, 2), (2, 2), (2, 2)], latentSpaceLength=128, controllerNodesPerLayer=[64, 32], controllerOutputLength=len(outputs), rewardNodesPerLayer=[16, 16], rewardOutputLength=2)
    while True:
        screen = R3U.takeScreenShot(0, 0, 1920, 1080, imageSize)
        screen = np.reshape(screen, newshape=(1, screen.shape[1], screen.shape[0], screen.shape[2]))
        output = controller.predict(screen)
        KeyOut.performOutput(OC.convertOutputToChar(output, outputs))

time.sleep(2)
testGameplay((360, 200), ['w', 'a', 's', 'd'])