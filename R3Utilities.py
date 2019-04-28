import time
from ctypes import *
from ctypes.wintypes import *
from PIL import Image
import numpy as np
import pyscreenshot as ImageGrab

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
        #print(char, tempIndex/length, (tempIndex+1)/length, percentage)
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


def readMemoryAddress(processID=4044, memoryAddress=0x1000000):
    OpenProcess = windll.kernel32.OpenProcess
    ReadProcessMemory = windll.kernel32.ReadProcessMemory
    CloseHandle = windll.kernel32.CloseHandle
    PROCESS_ALL_ACCESS = 0x1F0FFF
    pid = processID
    address = memoryAddress

    buffer = ctypes.c_ulong()
    bufferSize = ctypes.sizeof(buffer)
    bytesRead = c_ulong(0)


    processHandle = OpenProcess(PROCESS_ALL_ACCESS, False, pid)
    if ReadProcessMemory(processHandle, address, ctypes.byref(buffer), bufferSize, ctypes.byref(bytesRead)):
        print("Success:", buffer.value)
    else:
        print("Failed.")

    CloseHandle(processHandle)


def readMemoryTest():
    readMemoryAddress(processID=17928, memoryAddress=0x04311148)


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

def createNHotArray(arrayLength, indices):
    ret = []
    for i in range(arrayLength):
        if i in indices:
            ret.append(1.0)
        else:
            ret.append(0.0)
    return ret