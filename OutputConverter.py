import numpy as np


def convertOutputToChar(inputArray, outputArray):
    return outputArray[np.argmax(inputArray)]


def convertCharToHex(char):
    dict = {
        'q':0x10, 'w':0x11, 'e':0x12, 'r':0x13, 't':0x14, 'y':0x15, 'u':0x16, 'i':0x17, 'o':0x18, 'p':0x19,
        'a': 0x1E, 's': 0x1F, 'd': 0x20, 'f': 0x21, 'g': 0x22, 'h': 0x23, 'j': 0x24, 'k': 0x25, 'l': 0x26,
        'z': 0x2c, 'x': 0x2d, 'c': 0x2e, 'v': 0x2f, 'b': 0x30, 'n': 0x31, 'm': 0x32
    }
    return dict[char]


def convertOutputToHex(inputArray, outputArray):
    x = convertOutputToChar(inputArray, outputArray)
    return convertCharToHex(x)
