import time
from ctypes import *
from ctypes.wintypes import *
from PIL import Image
import numpy as np
import random
import KeyboardOutput as KeyOut
import pyscreenshot as ImageGrab
import ctypes
import math


def takeScreenShot(X1, Y1, X2, Y2, outputSize):
    im = ImageGrab.grab(bbox=(X1, Y1, X2, Y2), childprocess=False)
    im = im.resize(outputSize, Image.ANTIALIAS)
    im = np.array(im)
    return im


def takeScreenShotTest():
    image = takeScreenShot(0, 0, 1920, 1080)
    image.show()



def printLoadBar (percentage, length, endString=""):
    bar = "["
    for tempIndex in range(length):
        char = "-"
        if tempIndex/length <= percentage:
            if (tempIndex+1)/length > percentage:
                char = ">"
            else:
                char = "="
        bar += char
    bar += "]"
    if percentage == 0:
        print("%d%s%s%s" % ((percentage * 100), "%", bar, endString), end="")
    elif percentage >= 1:
        print("%s%d%s%s%s" % ("\r", (percentage * 100), "%", bar, endString))
    else:
        print("%s%d%s%s%s" % ("\r", (percentage * 100), "%", bar, endString), end="")


def printLoadBarTest():
    for i in range(100):
        printLoadBar(i/100, 20)
        time.sleep(0.1)
    printLoadBar(1, 20)


def readMemoryAddress(processID=4044, memoryAddress=0x1000000, bufferType=c_int32):
    OpenProcess = windll.kernel32.OpenProcess
    ReadProcessMemory = windll.kernel32.ReadProcessMemory
    CloseHandle = windll.kernel32.CloseHandle
    PROCESS_ALL_ACCESS = (0x000F0000 | 0x00100000 | 0xFFF)
    PROCESS_VM_READ = 0x0010
    pid = processID
    address = memoryAddress

    buffer = bufferType()
    #buffer = c_wchar_p("Test Test Test")
    #bufferSize = len(buffer.value)
    bufferSize = ctypes.sizeof(buffer)
    bytesRead = bufferType()

    processHandle = OpenProcess(PROCESS_VM_READ, False, pid)
    ReadProcessMemory(processHandle, c_void_p(address), ctypes.byref(buffer), bufferSize, ctypes.byref(bytesRead))
    #if ReadProcessMemory(processHandle, c_void_p(address), ctypes.byref(buffer), bufferSize, ctypes.byref(bytesRead)):
        #print("Success:", buffer.value) awa  f       fffff
    #else:
        #print("Failed.")

    CloseHandle(processHandle)
    return buffer.value


def readMemoryTest():
    readMemoryAddress(processID=17928, memoryAddress=0x04311148)


def convertOutputToChar(inputArray, outputArray):
    outputIndex = np.argmax(inputArray)
    return outputArray[outputIndex]


def convertCharToHex(char):
    dict = {
        'q':0x10, 'w':0x11, 'e':0x12, 'r':0x13, 't':0x14, 'y':0x15, 'u':0x16, 'i':0x17, 'o':0x18, 'p':0x19,
        'a': 0x1E, 's': 0x1F, 'd': 0x20, 'f': 0x21, 'g': 0x22, 'h': 0x23, 'j': 0x24, 'k': 0x25, 'l': 0x26,
        'z': 0x2c, 'x': 0x2d, 'c': 0x2e, 'v': 0x2f, 'b': 0x30, 'n': 0x31, 'm': 0x32,
        "space": 0x39, "lshift": 0x2A, "lctrl": 0x1D, "lalt": 0x38, "tab":0x0F, "lmouse": 0x02, "rmouse": 0x08, "midmouse": 0x20,
        "up": 0xC8, "left": 0xCB, "right": 0xCD, "down": 0xD0, 'none': -1
    }
    mouse = False
    if char == "lmouse" or char == "rmouse" or char == "midmouse":
        mouse = True
    return dict[char], mouse


def convertOutputToHex(inputArray, outputArray):
    x = convertOutputToChar(inputArray, outputArray)
    return convertCharToHex(x)


def createNHotArray(arrayLength, indices, value=1.0):
    ret = np.zeros(shape=(1,))
    for i in range(arrayLength):
        if i in indices:
            ret = np.append(ret, [value], axis=0)
        else:
            ret = np.append(ret, [0.0], axis=0)
        if i == 0:
            ret = np.delete(ret, 0)
    return ret


def generateQArray(rawRewards, rewardFunction, discountRate=0.9, speedUpRate=1, verbose=False):
    q = None
    print("Generating Q Array")
    for i in range(len(rawRewards)):
        if i % speedUpRate == 0:
            sum = 0
            lookAheadLen = math.log(0.001, discountRate) #limit how far the qArray looks ahead for computational efficiency
            for j in range(i, min(int(i + lookAheadLen + 1), len(rawRewards))):
                sum = sum + rewardFunction(rawRewards, j) * (discountRate ** (j - i))
            if q is None:
                q = np.array([sum])
            else:
                q = np.append(q, [sum], axis=0)
        printLoadBar((i + 1)/len(rawRewards), 25)
    if verbose:
        print(q)
    return q


def calculateQValue(rawRewards, rewardFunction, discountRate=0.9):
    sum = 0
    for i in range(len(rawRewards)):
        sum = sum + rewardFunction(rawRewards, i) * (discountRate ** i)
    return sum

def performOutput(outputChar, previousOutputChar):
    outputHex, isMouse = convertCharToHex(outputChar)
    previousOutput, _ = convertCharToHex(previousOutputChar)
    KeyOut.performOutput(outputHex, isMouse, previousOutput)
    if isMouse:
        return previousOutputChar
    else:
        return outputChar

def stringToType(inputType):
    conversion = {
        "float":ctypes.c_float, "int":ctypes.c_int32
    }
    return conversion[inputType.lower()]