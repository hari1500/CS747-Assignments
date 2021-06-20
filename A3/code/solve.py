import numpy as np
import matplotlib.pyplot as plt

seeds, algorithm, episodes, stochastic, mdp, task = None, None, None, None, None, None

""" stores MDP related data """
class MDP:
    def __init__(self, actions=4, epsilon=0.1, alpha=0.5, gamma=1.0):
        self.rows   = 7
        self.cols   = 10
        self.start  = (3, 0)
        self.end    = (3, 7)
        self.wind   = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.actions= actions
        self.reward = -1
        self.epsilon= epsilon
        self.alpha  = alpha
        self.gamma  = gamma

""" converts state from tuple to index """
def getIndFromTuple(stateTuple, mdp):
    return stateTuple[0] * mdp.cols + stateTuple[1]

""" converts state from index to tuple """
def getTupleFromInd(stateInd, mdp):
    return (stateInd // mdp.cols, stateInd % mdp.cols)

""" Setting up random seed """
def setupSeed(seed=0):
    np.random.seed(seed)

""" each episode """
def runEpisode(mdp, Q, algorithm="sarsa", stochastic=False):
    def getAction(stateInd, Q):
        if (np.random.binomial(1, mdp.epsilon) == 1):            
            return np.random.randint(mdp.actions)
        return np.argmax(Q[stateInd])

    def getNewState(currStateInd, currAction):
        row, col = getTupleFromInd(currStateInd, mdp)
        row += mdp.wind[col]
        if stochastic:
            row += np.random.choice([-1,0,1])

        """
            Up      North       0
            Right   East        1
            Down    South       2
            Left    West        3
                    North-East  4
                    South-East  5
                    South-West  6
                    North-West  7
        """
        if (currAction in [0, 4, 7]):
            row += 1
        if (currAction in [1, 4, 5]):
            col += 1
        if (currAction in [2, 5, 6]):
            row -= 1
        if (currAction in [3, 6, 7]):
            col -= 1

        row = max(min(row, mdp.rows-1), 0)
        col = max(min(col, mdp.cols-1), 0)

        return getIndFromTuple((row, col), mdp)

    # Checking for algorithm
    algorithm = str(algorithm).lower()
    if (algorithm not in ['sarsa', 'esarsa', 'qlearning']):
        print("algorithm should be sarsa or qlearning or esarsa")
        exit()

    # Actual Episode running
    currState   = getIndFromTuple(mdp.start, mdp)
    termState   = getIndFromTuple(mdp.end, mdp)
    currAction  = getAction(currState, Q)

    steps = 0
    while currState != termState:
        newState    = getNewState(currState, currAction)
        newAction   = getAction(newState, Q)
        reward      = 0 if newState == termState else mdp.reward
        
        target      = 0
        if (algorithm == "sarsa"):
            target = reward + mdp.gamma * Q[newState][newAction]
        elif (algorithm == "qlearning"):
            target = reward + mdp.gamma * np.max(Q[newState])
        elif (algorithm == "esarsa"):
            maxQ = np.max(Q[newState])
            expectedQ = 0
            uniformProb = mdp.epsilon / mdp.actions
            greedyProb = (1 - mdp.epsilon) / (Q[newState] == maxQ).sum()
            for q in Q[newState]:
                expectedQ += q * (uniformProb + (greedyProb if q == maxQ else 0))
            target = reward + mdp.gamma * expectedQ

        Q[currState][currAction] += mdp.alpha * (target - Q[currState][currAction])

        currState   = newState
        currAction  = newAction
        steps       += 1
        # print(getTupleFromInd(currState, mdp), currAction)

    return Q, steps

""" returns time steps for episodes """
def getTimeSteps(mdp, episodes, seeds, algorithm="sarsa", stochastic=False):
    timeSteps = np.zeros(episodes)
    for seed in range(seeds):
        # print("seed", seed)
        setupSeed(seed)
        Q = np.zeros((mdp.rows * mdp.cols, mdp.actions))
        episodicSteps = []
        for _ in range(episodes):
            Q, steps = runEpisode(mdp, Q, algorithm=algorithm, stochastic=stochastic)
            episodicSteps.append(steps)
        timeSteps += np.cumsum(episodicSteps)
    timeSteps /= seeds

    return timeSteps

""" Task 2 """
def runTask2():
    global episodes, mdp, seeds
    print("Plotting Task 2")
    mdp.actions = 4

    plt.close()
    plt.xlabel("Time Steps")
    plt.ylabel("Episodes")
    plt.plot(getTimeSteps(mdp, episodes, seeds), list(range(1, episodes + 1)), label="Sarsa(0)")
    plt.legend()
    plt.savefig('task-2.png')
    plt.close()

""" Task 3 """
def runTask3():
    global episodes, mdp, seeds
    print("Plotting Task 3")
    mdp.actions = 8

    plt.close()
    plt.xlabel("Time Steps")
    plt.ylabel("Episodes")
    plt.plot(getTimeSteps(mdp, episodes, seeds), list(range(1, episodes + 1)), label="King's moves")
    plt.legend()
    plt.savefig('task-3.png')
    plt.close()

""" Task 4 """
def runTask4():
    global episodes, mdp, seeds
    print("Plotting Task 4")
    mdp.actions = 4

    plt.close()
    plt.xlabel("Time Steps")
    plt.ylabel("Episodes")
    plt.plot(getTimeSteps(mdp, episodes, seeds, stochastic=True), list(range(1, episodes + 1)), label="Stochastic")
    plt.legend()
    plt.savefig('task-4.png')
    plt.close()

""" Task 5 """
def runTask5():
    global episodes, mdp, seeds
    print("Plotting Task 5")
    mdp.actions = 4

    plt.close()
    plt.xlabel("Time Steps")
    plt.ylabel("Episodes")
    plt.plot(getTimeSteps(mdp, episodes, seeds, "sarsa"), list(range(1, episodes + 1)), label="Sarsa")
    plt.plot(getTimeSteps(mdp, episodes, seeds, "qlearning"), list(range(1, episodes + 1)), label="Q-Learning")
    plt.plot(getTimeSteps(mdp, episodes, seeds, "esarsa"), list(range(1, episodes + 1)), label="Expected Sarsa")
    plt.legend()
    plt.savefig('task-5.png')
    plt.close()

""" Parses input arguments """
def parseArgs():
    import argparse
    global seeds, mdp, algorithm, task, episodes, stochastic
    def parseArg(arg, type, errStmt=""):
        try:
            if (type == "s" or type == "epi"):
                arg = int(arg)
                if (arg >= 0): return arg
            if (type == "act"):
                arg = int(arg)
                if (arg == 4 or arg == 8): return arg
            if (type == "alg"):
                arg = str(arg).lower()
                if (arg in ['sarsa', 'esarsa', 'qlearning']): return arg
            if (type == "t"):
                arg = int(arg)
                if (arg in [0, 2, 3, 4, 5, 9]): return arg
            if (type == "eps" or type == "alp"):
                arg = float(arg)
                if (arg > 0 and arg <= 1): return arg
            if (type == "g"):
                arg = float(arg)
                if (arg >= 0 and arg <= 1): return arg
            if (type == "st"):
                arg = str(arg).lower()
                if (arg == 'true'): return True
                if (arg == 'false'): return False

            print(errStmt)
            exit()
        except Exception:
            print(errStmt)
            exit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", "-s", default=100)
    parser.add_argument("--algorithm", "-alg", default="sarsa")
    parser.add_argument("--episodes", "-epi", default=200)
    parser.add_argument("--stochastic", "-st", default=False)
    parser.add_argument("--task", "-t", default=0)
    parser.add_argument("--actions", "-act", default=4)
    parser.add_argument("--epsilon", "-eps", default=0.1)
    parser.add_argument("--alpha", "-alp", default=0.5)
    parser.add_argument("--gamma", "-g", default=1)
    args = parser.parse_args()

    seeds       = parseArg(args.seeds, "s", "--seeds should be an integer")
    algorithm   = parseArg(args.algorithm, "alg", "--algorithm should be sarsa or qlearning or esarsa")
    episodes    = parseArg(args.episodes, "epi", "--episodes should be an integer")
    stochastic  = parseArg(args.stochastic, "st", "--stochastic should be true or false")
    actions     = parseArg(args.actions, "act", "--actions should be 4 or 8")
    task        = parseArg(args.task, "t", "--task should be one of [0, 2, 3, 4, 5, 9]")
    epsilon     = parseArg(args.epsilon, "eps", "--epsilon should be in between 0 and 1 (both excluded)")
    alpha       = parseArg(args.alpha, "alp", "--alpha should be in between 0 and 1 (both excluded)")
    gamma       = parseArg(args.gamma, "g", "--gamma should be in between 0 and 1 (both included)")

    mdp = MDP(actions, epsilon, alpha, gamma)
    return


if __name__ == "__main__":
    # Reads input arguments
    parseArgs()

    if (task == 9):
        setupSeed(seeds)
        Q = np.zeros((mdp.rows * mdp.cols, mdp.actions))
        Q, steps = runEpisode(mdp, Q, algorithm, stochastic)
        print(steps)

    if (task == 0 or task == 2): runTask2()
    if (task == 0 or task == 3): runTask3()
    if (task == 0 or task == 4): runTask4()
    if (task == 0 or task == 5): runTask5()