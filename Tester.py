import time
import R3Utilities as R3Util
import ConvolutionalAutoencoder as CAE
import numpy
import random
import tensorflow as tf
from PIL import Image

def encodeScreenshotTest(x1, x2, y1, y2, originalSize, compressedSize, numOfImages):
    inData = []

    inShape = (compressedSize[0], compressedSize[1], 3)
    enc, dec, cnae = CAE.generateNetworks(inShape, [64, 48, 48], [(6, 6), (4, 4), (3, 3)], 256, hiddenActivation=tf.nn.leaky_relu, learningRate=0.0001, useDroput=False)

    for i in range(numOfImages):
        randX = random.randint(x1, x2)
        randY = random.randint(y1, y2)
        im = R3Util.takeScreenShot(randX, randY, randX + originalSize[0], randY + originalSize[1])
        im = im.resize(compressedSize, Image.ANTIALIAS)
        im = numpy.array(im)
        im = im/256
        inData.append(im)
        R3Util.printLoadBar((i+1)/numOfImages, 50)
    inData = numpy.reshape(inData, newshape=(len(inData), compressedSize[0], compressedSize[1], 3))
    CAE.trainCNAE(cnae, inData, batchSize=1, iterations=5, verbose=True)
    CAE.testCNAE(enc, dec, inData, batchSize=1, displayedRows=5, displayedColumns=5)


encodeScreenshotTest(0, 840, 0, 0, (1080, 1080), (240, 240), 1000)