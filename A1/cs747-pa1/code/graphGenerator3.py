import numpy as np
import matplotlib.pyplot as plt

data = []
with open('outputDataT3.txt', 'r') as f:
    data = [line.strip().split(", ") for line in f]
    data = np.asarray(data)

def getAvgRegrets(data, instance):
    nData = data[(data[:,0] == instance)]
    regrets = {}
    for row in nData:
        eps = float(row[3])
        reg = float(row[5])
        if eps not in regrets.keys():
            regrets[eps] = 0
        regrets[eps] += reg

    retDict = {}
    for epsilon in regrets.keys():
        print("%s, %s, %s" % (instance, epsilon, regrets[epsilon]/50))
        retDict[epsilon] = regrets[epsilon] / 50

    return retDict

r1 = getAvgRegrets(data, "../instances/i-1.txt")

plt.plot(r1.keys(), r1.values())
plt.title('T3: ../instances/i-1.txt')
plt.ylabel('REGRET')
plt.xlabel('Epsilon')
plt.savefig("3_1.png")
plt.close()

r2 = getAvgRegrets(data, "../instances/i-2.txt")

plt.plot(r2.keys(), r2.values())
plt.title('T3: ../instances/i-2.txt')
plt.ylabel('REGRET')
plt.xlabel('Epsilon')
plt.savefig("3_2.png")
plt.close()

r3 = getAvgRegrets(data, "../instances/i-3.txt")

plt.plot(r3.keys(), r3.values())
plt.title('T3: ../instances/i-3.txt')
plt.ylabel('REGRET')
plt.xlabel('Epsilon')
plt.savefig("3_3.png")
plt.close()
