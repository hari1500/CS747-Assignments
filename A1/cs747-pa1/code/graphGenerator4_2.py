import numpy as np
import matplotlib.pyplot as plt

data = []
with open('outputDataT2.txt', 'r') as f:
    data = [line.strip().split(", ") for line in f]
    data = np.asarray(data)

def getAvgRegrets(data, instance, algorithm):
    nData = data[(data[:,0] == instance) & (data[:,1] == algorithm)]
    regrets = {}
    for row in nData:
        hor = int(row[4])
        reg = float(row[5])
        if hor not in regrets.keys():
            regrets[hor] = 0
        regrets[hor] += reg

    retDict = {}
    for horizon in regrets.keys():
        print("%s, %s, %s, %s" % (instance, algorithm, horizon, regrets[horizon]/50))
        retDict[np.log10(horizon)] = regrets[horizon] / 50

    return retDict

r1_1 = getAvgRegrets(data, "../instances/i-1.txt", "thompson-sampling")
r1_2 = getAvgRegrets(data, "../instances/i-1.txt", "thompson-sampling-with-hint")

plt.plot(r1_1.keys(), r1_1.values(), 'r-', label="Thompson Sampling")
plt.plot(r1_2.keys(), r1_2.values(), 'g-', label="Thompson Sampling With Hint")
plt.title('T2: ../instances/i-1.txt')
plt.ylabel('REGRET')
plt.xlabel('log of HORIZON')
plt.legend()
plt.savefig("4_2_1.png")
plt.close()

r2_1 = getAvgRegrets(data, "../instances/i-2.txt", "thompson-sampling")
r2_2 = getAvgRegrets(data, "../instances/i-2.txt", "thompson-sampling-with-hint")

plt.plot(r2_1.keys(), r2_1.values(), 'r-', label="Thompson Sampling")
plt.plot(r2_2.keys(), r2_2.values(), 'g-', label="Thompson Sampling With Hint")
plt.title('T2: ../instances/i-2.txt')
plt.ylabel('REGRET')
plt.xlabel('log of HORIZON')
plt.legend()
plt.savefig("4_2_2.png")
plt.close()

r3_1 = getAvgRegrets(data, "../instances/i-3.txt", "thompson-sampling")
r3_2 = getAvgRegrets(data, "../instances/i-3.txt", "thompson-sampling-with-hint")

plt.plot(r3_1.keys(), r3_1.values(), 'r-', label="Thompson Sampling")
plt.plot(r3_2.keys(), r3_2.values(), 'g-', label="Thompson Sampling With Hint")
plt.title('T2: ../instances/i-3.txt')
plt.ylabel('REGRET')
plt.xlabel('log of HORIZON')
plt.legend()
plt.savefig("4_2_3.png")
plt.close()
