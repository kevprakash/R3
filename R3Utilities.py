import time
from ctypes import *
from ctypes.wintypes import *

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


#printLoadBarTest()
readMemoryTest()