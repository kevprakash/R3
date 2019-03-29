import time
import pyscreenshot as ImageGrab

# takes screenshot of part of the screen

#may not be the best labels or inccorect
def takeScreenShot(X1, Y1, X2, Y2):
    im = ImageGrab.grab(bbox=(X1, Y1, X2, Y2), childprocess=False)
    return im

def takeScreenShotTest():
    image = takeScreenShot(0, 0, 200, 200)
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


#printLoadBarTest()
takeScreenShotTest()