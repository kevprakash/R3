import tkinter as tk
from tkinter import ttk
import importlib
import R3Utilities as util
import MonolithNetwork as mn
import Runner
import time

model = None


def addLabel(addTo, text, x, y, font="Verdana 8", relief="flat", padx=(10, 10), pady=(10, 10)):
    label = tk.Label(addTo, text=text, font=font, relief=relief)
    label.grid(column=x, row=y, padx=padx, pady=pady)
    return label


def addInput(addTo, height, width, x, y, padx=(10, 10), pady=(10, 10)):
    textBox = tk.Text(addTo, height=height, width=width, wrap=tk.WORD)
    textBox.grid(column=x, row=y, padx=padx, pady=pady, sticky=tk.N)
    return textBox


def addLabeledInput(addTo, text, inputHeight, width, x, y, font="Verdana 8", relief="flat", padx=(10, 10), pady=(10, 10)):
    addLabel(addTo, text, x, y, font=font, relief=relief, padx=padx, pady=(pady[0], 1))
    textBox = addInput(addTo, inputHeight, width, x, y + 1, padx=padx, pady=(1, pady[1]))
    return textBox


def addButton(addTo, text, height, width, x, y, command, padx=(10, 10), pady=(10, 10)):
    button = tk.Button(addTo, text=text, height=height, width=width, command=lambda: command())
    button.grid(column=x, row=y, padx=padx, pady=pady)
    return button


def addTab(control, name):
    tab = tk.Frame(control)
    control.add(tab, text=name)
    control.pack(expand=True, fill="both")
    return tab


def addCheckbox(addTo, text, x, y, height=0, width=0, padx=(10, 10), pady=(10, 10)):
    v = tk.IntVar()
    checkbox = tk.Checkbutton(addTo, text=text, variable=v, height=height, width=width)
    checkbox.grid(column=x, row=y, padx=padx, pady=pady)
    return checkbox, v


def importModuleFunction(moduleName, functionName, filePath):
    module_spec = importlib.util.spec_from_file_location(moduleName, filePath)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, functionName)

window = tk.Tk()
window.title("Recurrent Reward-Learning Reinforcement")

#Size of the window
width_of_window = 769
height_of_window = 588

#Depends on what DPI are you using
screen_width = window.winfo_screenwidth()
screen_heigth = window.winfo_screenheight()

#Calculate x and y coordinate
x_coordinate = (screen_width / 2) - (width_of_window / 2)
y_coordinate = (screen_heigth / 2) - (height_of_window / 2)

#Display the window in the screen
window.geometry("%dx%d+%d+%d" % (width_of_window, height_of_window, x_coordinate, y_coordinate))
window.resizable(width=False, height=False)

#Create Tab Control
tabControl = ttk.Notebook(window)

networkTab = addTab(tabControl, "Network Settings")
gameTab = addTab(tabControl, "Game Settings")
runningTab = addTab(tabControl, "Running/Training Settings")
start = addTab(tabControl, "Start")

#Network Settings
convFilters = addLabeledInput(networkTab, "Number of Convolution Filters", 1.35, 40, 0, 0, font="Verdana 12", pady=(10, 0))
convKernels = addLabeledInput(networkTab, "Size of Convolution Filters", 1.35, 40, 1, 0, font="Verdana 12", pady=(10, 0))
convStrides = addLabeledInput(networkTab, "Convolution Strides", 1.35, 40, 0, 2, font="Verdana 12", pady=(10, 0))
hiddenNodes = addLabeledInput(networkTab, "Number of Hidden Nodes", 1.35, 40, 1, 2, font="Verdana 12", pady=(10, 0))
memoryLength = addLabeledInput(networkTab, "Memory Length", 1.35, 40, 0, 4, font="Verdana 12", pady=(10, 0))
recurrenceLength = addLabeledInput(networkTab, "Recurrence Length", 1.35, 40, 1, 4, font="Verdana 12", pady=(10, 0))

#Game Settings
gamePID = addLabeledInput(gameTab, "Process ID", 1.35, 40, 0, 0, font="Verdana 12", pady=(10, 0))
gameAddrs = addLabeledInput(gameTab, "Memory Addresses", 20, 40, 0, 2, font="Verdana 12", pady=(10, 0))
screenBoundingBox = addLabeledInput(gameTab, "Screen Bounding Box", 1.35, 40, 1, 0, font="Verdana 12", pady=(10, 0))
scaledResolution = addLabeledInput(gameTab, "Scaled Resolution", 1.35, 40, 1, 2, font="Verdana 12", pady=(10, 0))

#Running/Training Settings
rewardFunctionScript = addLabeledInput(runningTab, "Reward Function Script", 1.35, 40, 0, 0, font="Verdana 12", pady=(10, 0))
rewardFunctionPath = addLabeledInput(runningTab, "Reward Function Path", 1.35, 40, 1, 0, font="Verdana 12", pady=(10, 0))
rewardFunctionName = addLabeledInput(runningTab, "Reward Function Name", 1.35, 40, 0, 2, font="Verdana 12", pady=(10, 0))
shouldTrainCheckbox, shouldTrain = addCheckbox(runningTab, "Train", 1, 3)

keyboardOutput = addLabeledInput(runningTab, "Keyboard Outputs", 3, 40, 0, 4, font="Verdana 12", pady=(10, 0))
killSwitch = addLabeledInput(runningTab, "Kill Switch Key", 1.35, 40, 1, 4, font="Verdana 12", pady=(10, 0))

learningRate = addLabeledInput(runningTab, "Learning Rate", 1.35, 40, 0, 6, font="Verdana 12", pady=(10, 0))
explorationRate = addLabeledInput(runningTab, "Base Exploration Rate", 1.35, 40, 1, 6, font="Verdana 12", pady=(10, 0))


def generateModel():
    temp = convFilters.get('1.0', tk.END).split(", ")
    convFiltersConverted = []
    for i in temp:
        convFiltersConverted.append(int(i))

    temp = convKernels.get('1.0', tk.END).split(", ")
    convKernelsConverted = []
    for i in temp:
        convKernelsConverted.append((int(i), int(i)))

    temp = convStrides.get('1.0', tk.END).split(", ")
    convStridesConverted = []
    for i in temp:
        convStridesConverted.append((int(i), int(i)))

    temp = hiddenNodes.get('1.0', tk.END).split(", ")
    hiddenNodesConverted = []
    for i in temp:
        hiddenNodesConverted.append(int(i))

    temp = scaledResolution.get('1.0', tk.END).split(", ")
    scaledResolutionConverted = (int(temp[0]), int(temp[1]))

    outputLen = len(keyboardOutput.get('1.0', tk.END).split(", "))

    global model
    model = mn.generateNetwork(scaledResolutionConverted, convFiltersConverted, convKernelsConverted,
                               convStridesConverted, hiddenNodesConverted, outputLen)


def run():
    time.sleep(2)
    memoryLengthConverted = int(memoryLength.get('1.0', tk.END))
    recurrenceLengthConverted = int(recurrenceLength.get('1.0', tk.END))

    gamePIDConverted = int(gamePID.get('1.0', tk.END), 16)
    temp = gameAddrs.get('1.0', tk.END).split("\n")
    temp = temp[:len(temp) - 1]
    gameAddrsConverted = []
    gameAddrsType = []
    for i in temp:
        x = i.split(":")
        addr = x[0]
        addrType = x[1]
        addrType = util.stringToType(addrType)
        gameAddrsConverted.append(int(addr, 16))
        gameAddrsType.append(addrType)

    temp = screenBoundingBox.get('1.0', tk.END).split(", ")
    screenCorners = [(int(temp[0]), int(temp[1])), (int(temp[2]), int(temp[3]))]

    temp = scaledResolution.get('1.0', tk.END).split(", ")
    scaledResolutionConverted = (int(temp[0]), int(temp[1]))

    rewardFunction = importModuleFunction(rewardFunctionScript.get('1.0', tk.END)[:-1],
                                          rewardFunctionName.get('1.0', tk.END)[:-1],
                                          rewardFunctionPath.get('1.0', tk.END)[:-1])

    temp = keyboardOutput.get('1.0', tk.END)[:-1].split(", ")
    keyboardOutputConverted = []
    for i in temp:
        keyboardOutputConverted.append(i)

    killSwitchConverted = killSwitch.get('1.0', tk.END)[:-1]

    learningRateConverted = float(learningRate.get('1.0', tk.END))
    explorationRateConverted = float(explorationRate.get('1.0', tk.END))

    global model
    Runner.runAndTrain(screenCorners[0], screenCorners[1], scaledResolutionConverted, model, keyboardOutputConverted,
                       gamePIDConverted, gameAddrsConverted, gameAddrsType, rewardFunction, memoryLengthConverted,
                       recurrenceLengthConverted, killSwitch=killSwitchConverted, train=shouldTrain.get(),
                       qLearningRate=learningRateConverted, explorationRate=explorationRateConverted)

    temp = keyboardOutput.get('1.0', tk.END).split(", ")
    keyboardOutputConverted = []
    for i in temp:
        keyboardOutputConverted.append(i)


#Start Tab
modelButton = addButton(start, "Generate Model", 5, 25, 0, 0, generateModel)
startButton = addButton(start, "Start", 5, 25, 1, 0, run)

window.mainloop()