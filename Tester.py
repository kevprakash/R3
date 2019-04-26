import time
import R3Utilities as R3Util
import ConvolutionalAutoencoder as CAE
import RewardNetwork as RN
import numpy
import random
import tensorflow as tf
from PIL import Image

def encodeScreenshotTest(x1, x2, y1, y2, originalSize, compressedSize, numOfImages, trainingIterations=10, batchSize=16, timeBetweenScreenshots=0):
    inShape = (compressedSize[0], compressedSize[1], 3)
    enc, dec, cnae = CAE.generateNetworks(inShape, [64, 48, 48], [(6, 4), (4, 4), (3, 3)], 128, hiddenActivation=tf.nn.leaky_relu, learningRate=0.0001, dropoutRate=0)

    inData = []
    for i in range(numOfImages):
        randX = random.randint(x1, x2)
        randY = random.randint(y1, y2)
        im = R3Util.takeScreenShot(randX, randY, randX + originalSize[0], randY + originalSize[1], compressedSize)
        im = numpy.array(im)
        im = im/256
        inData.append(im)
        R3Util.printLoadBar((i+1)/numOfImages, 50)
        if timeBetweenScreenshots > 0:
            time.sleep(timeBetweenScreenshots)
    inData = numpy.reshape(inData, newshape=(len(inData), compressedSize[0], compressedSize[1], 3))
    CAE.trainCNAE(cnae, inData, batchSize=batchSize, iterations=trainingIterations, verbose=True)
    CAE.testCNAE(enc, dec, inData, batchSize=1, displayedRows=5, displayedColumns=5)

#time.sleep(1)
#encodeScreenshotTest(0, 0, 0, 0, (1920, 1080), (288, 192), 1000, batchSize=16, timeBetweenScreenshots=0)