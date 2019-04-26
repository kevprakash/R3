import numpy as np

def convertOutputToChar(inputArray, outputArray):
    return outputArray[np.argmax(inputArray)]