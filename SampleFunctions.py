import math

def sampleRewardFunction(rawInputs, index):
    raw = rawInputs[index]
    prevRaw = rawInputs[index - 1]
    if index == 0 or (raw[3] == 0 and raw[4] == 0):
        return 0
    if raw[1] <= 0:
        if prevRaw[1] > 0:
            return -100
        else:
            return 0
    reward = 0
    loc = math.sqrt((raw[3] ** 2) + (raw[4] ** 2))
    prevLoc = math.sqrt((prevRaw[3] ** 2) + (prevRaw[4] ** 2))

    reward = reward + min((math.fabs(loc - prevLoc) - 2) * 2, 5)

    if prevRaw[1] > 0:
        reward = reward + ((raw[1] + raw[2]) - (prevRaw[1] + prevRaw[2])) * 20
    reward = reward + (raw[0] - prevRaw[0]) * 50

    return reward
