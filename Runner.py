import MonolithNetwork as mn
import tensorflow as tf
import R3Utilities as util
import numpy as np
import random
import ctypes
from tensorflow import keras
import SampleFunctions as sf
import keyboard

def runAndTrain(topLeft, bottomRight, scaledSize, model, outputs, processID, rewardMemoryAddresses, rewardType, rewardFunction, lookAheadLength, recurrenceLength, killSwitch='p', train=True, qLearningRate=0.5, explorationRate=0.25):
    print("Setting up for Run")
    startUp = 0
    if train:
        memorySequence = np.zeros(shape=(1, lookAheadLength + recurrenceLength, scaledSize[1], scaledSize[0], 3))
        actionPredicted = np.zeros(shape=(lookAheadLength + 1, len(outputs)))
        actionsTaken = np.zeros(shape=(lookAheadLength + 1,))
        rewards = np.zeros(shape=(lookAheadLength, len(rewardMemoryAddresses)))

    recurrenceSequence = np.zeros(shape=(1, recurrenceLength, scaledSize[1], scaledSize[0], 3))

    previousOutput = "none"
    print("Beginning Run")
    while not keyboard.is_pressed(killSwitch):
        screen = util.takeScreenShot(topLeft[0], topLeft[1], bottomRight[0], bottomRight[1], scaledSize)

        recurrenceSequence = np.append(recurrenceSequence, [[screen]], axis=1)
        recurrenceSequence = np.delete(recurrenceSequence, 0, axis=1)

        predictions = model.predict_on_batch(recurrenceSequence)

        if random.uniform(0, 1) < explorationRate:
            outputIndex = random.randint(0, len(outputs) - 1)
        else:
            outputIndex = np.argmax(predictions[0])

        output = outputs[outputIndex]

        previousOutput = util.performOutput(output, previousOutput)


        if train:
            memorySequence = np.append(memorySequence, [[screen]], axis=1)
            memorySequence = np.delete(memorySequence, 0, axis=1)

            actionPredicted = np.append(actionPredicted, [predictions[0]], axis=0)
            actionPredicted = np.delete(actionPredicted, 0, axis=0)

            actionsTaken = np.append(actionsTaken, [outputIndex], axis=0)
            actionsTaken = np.delete(actionsTaken, 0, axis=0)

            temp = []
            for i in range(len(rewardMemoryAddresses)):
                temp.append(util.readMemoryAddress(processID, rewardMemoryAddresses[i], rewardType[i]))

            rewards = np.append(rewards, [temp], axis=0)
            rewards = np.delete(rewards, 0, axis=0)

            if startUp < len(memorySequence[0]):
                startUp = startUp + 1
                print(output, " => ", temp, " => ", explorationRate)
            else:
                q = util.calculateQValue(rewards, rewardFunction, discountRate=0.9)
                update = []
                for i in range(len(outputs)):
                    if i == int(actionsTaken[0]):
                        update.append(actionPredicted[0][int(actionsTaken[0])] * (1 - qLearningRate) + qLearningRate * q)
                    else:
                        update.append(actionPredicted[0][i])

                if q < 0:
                    explorationRate = min(explorationRate * 1.01, 0.9)
                else:
                    explorationRate = max(explorationRate * 0.99, 0.1)

                subsequence = memorySequence[:1, :recurrenceLength]

                print(output, " => ", temp, " => ", q, " => ", explorationRate)

                model.train_on_batch(subsequence, np.array([update]))
    print("Finished Run")

'''
TESTING CODE 

imageSize = (360, 200)
keystrokes = list("wasdqe")
keystrokes.extend(["lmouse", "rmouse", "lshift", "space"])
PID = 0x215C
memoryAddresses = [0x9FF02118, 0x9FF020E4, 0x9FF020F0, 0x16D4A7A4, 0x16D4A7A8] #Geo, Health, Shield, xCoord, yCoord in Hollow Knight
types = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float]

model = mn.generateNetwork(imageSize=(imageSize[1], imageSize[0]),
                           numberOfFilters=[64, 32, 32], filterSizes=[(8, 8), (5, 5), (3, 3)], strideSizes=[(5, 5), (4, 4), (2, 2)],
                           hiddenNodes=[48, 32], outputLength=len(keystrokes))

runAndTrain((0, 0), (1920, 1080), imageSize, model, keystrokes, PID, memoryAddresses, types, sf.sampleRewardFunction, lookAheadLength=10, recurrenceLength=4)
'''