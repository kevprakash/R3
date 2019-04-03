import os
import R3Utilities as R3Util
import ConvolutionalAutoencoder as CAE
import numpy
import random
from PIL import Image

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def encodeScreenshotTest(x1, x2, y1, y2, originalSize, compressedSize, numOfImages):
    inData = []
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
    inShape = (compressedSize[0], compressedSize[1], 3)
    print(numpy.shape(inData))
    print(inShape)
    enc, dec = CAE.generateNetworks(inShape, [64, 64], [(4, 4), (4, 4)], useDroput=True)
    CAE.trainCNAE(enc, dec, inData, batchSize=1, iterations=5)
    CAE.testCNAE(enc, dec, inData, batchSize=1, displayedRows=5, displayedColumns=5)


encodeScreenshotTest(0, 0, 0, 0, (1920, 1080), (960, 540), 500)