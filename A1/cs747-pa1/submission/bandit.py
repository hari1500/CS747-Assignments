import numpy as np

# All possible ALGORITHMS
AL_EG   = "epsilon-greedy"
AL_UCB  = "ucb"
AL_KLUCB= "kl-ucb"
AL_TS   = "thompson-sampling"
AL_TSWH = "thompson-sampling-with-hint"

# All inputs global
banditInstance, algorithm, randomSeed, epsilon, horizon, REG = None, None, None, None, None, None
allowBreaks, breakPoints = False, [100, 400, 1600, 6400, 25600]

""" Printing 6-column format """
def printOutput(numIters=None):
    global banditInstance, algorithm, randomSeed, epsilon, horizon, REG
    if (banditInstance):
        if (numIters == None):
            numIters = horizon
        print("%s, %s, %s, %s, %s, %s" % (banditInstance.instanceFilePath, algorithm, randomSeed, epsilon, numIters, REG))

"""
    instanceFilePath    : string
    numArms             : integer
    meanRewards         : [ float ]
    highestMeanReward   : float
    empiricalResults    : [ [cum_reward-float, num_pulls-integer] ]
"""
class BanditInstance:
    def __init__(self, instanceFilePath, errStmt):
        try:
            self.instanceFilePath = instanceFilePath

            self.meanRewards = []
            with open(instanceFilePath, 'r') as f:
                lines = [line.strip() for line in f]
                for line in lines:
                    line = float(line)
                    if (line >= 0 and line <= 1):
                        self.meanRewards.append(line)
                    else:
                        raise ValueError()

            self.numArms = len(self.meanRewards)
            self.highestMeanReward = np.max(self.meanRewards)
            self.empiricalResults = [[0, 0] for _ in range(self.numArms)]
        except ValueError:
            print("Mean Rewards should lie in between 0 and 1")
            exit()
        except FileNotFoundError:
            print(errStmt)
            exit()
        except Exception as err:
            print(err)
            exit()
    
    def __str__(self):
        return "BANDIT INSTANCE: \n\tPath: %s \n\tNumber of arms: %s \n\tmean rewards: %s \n\thighest mean reward: %s \n\tempirical results: %s" % (self.instanceFilePath, self.numArms, self.meanRewards, self.highestMeanReward, self.empiricalResults)

    def pullArm(self, armIndex=-1):
        """ returns reward """
        if (armIndex < 0 or armIndex >= self.numArms):
            armIndex = np.random.randint(0, self.numArms)

        reward = self.genReward(armIndex)
        self.updateEmpiricalResults(armIndex, reward)
        return reward

    def genReward(self, armIndex=-1):
        """ Generates Reward """
        if (armIndex < 0 or armIndex >= self.numArms):
            return 0

        return np.random.binomial(1, self.meanRewards[armIndex])

    def updateEmpiricalResults(self, armIndex, reward):
        """ Updates Empirical Results """
        if (armIndex < 0 or armIndex >= self.numArms):
            return

        cumReward, numPulls = self.empiricalResults[armIndex]
        self.empiricalResults[armIndex] = [cumReward + reward, numPulls + 1]
        return

    def getEmpiricalMeans(self):
        """ returns Empirical means """
        empiricalMeans = []
        for result in self.empiricalResults:
            if (result[1] <= 0):
                empiricalMeans.append(0)
            else:
                empiricalMeans.append((result[0] * 1.0) / result[1])

        return empiricalMeans
    
    def getUCBs(self, t):
        """ returns UCBs at time t """
        ucbs = []
        for result in self.empiricalResults:
            if (result[1] <= 0):
                ucbs.append(0)
            else:
                ucbs.append( ((result[0] * 1.0) / result[1]) + np.sqrt(2 * np.log(t) / result[1]) )

        return ucbs

    def getKLUCBs(self, t, c):
        """ returns KLUCBs at time t """
        def getKLUCB(p_t, u_t):
            maxIter, precision = 50, 0.0001
            def KL(p, q):
                """ Finds KL divergence """
                if (q == p):
                    return 0
                if (q == 1):
                    return np.inf
                if (p == 0):
                    return np.log(1 - q)
                if (p == 1):
                    return np.log(q)
                return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


            numIter, l, u = 0, p_t, 1
            while (numIter < maxIter) and (u - l > precision):
                numIter += 1
                m = (u + l) / 2.0

                log_t = np.log(t)
                rhs = (log_t + c * np.log(log_t)) / u_t
                if (KL(p_t, m) < rhs):
                    l = m
                else:
                    u = m

            return (u + l) / 2.0

        kl_ucbs = []
        for result in self.empiricalResults:
            if (result[1] <= 0):
                kl_ucbs.append(0)
            else:
                kl_ucbs.append(getKLUCB(((result[0] * 1.0) / result[1]), result[1]))

        return kl_ucbs

    def getThompsonSamples(self):
        """ returns Thompson Samples """
        samples = []
        for result in self.empiricalResults:
            samples.append(np.random.beta(result[0] + 1, result[1] - result[0] + 1))

        return samples


""" Parses all input arguments """
def parseArgs():
    import argparse
    global banditInstance, algorithm, randomSeed, epsilon, horizon, allowBreaks
    def parseArg(arg, type, errStmt=""):
        try:
            if (type == "in"):
                return BanditInstance(arg, errStmt)
            elif (type == "al"):
                if (arg in [AL_EG, AL_UCB, AL_KLUCB, AL_TS, AL_TSWH]): return arg
            elif (type == "rs" or type == "hz"):
                arg = int(arg)
                if (arg >= 0): return arg
            elif (type == "ep"):
                arg = float(arg)
                if (arg >= 0 and arg <= 1): return arg

            print(errStmt)
            exit()
        except Exception:
            print(errStmt)
            exit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", "-in")
    parser.add_argument("--algorithm", "-al")
    parser.add_argument("--randomSeed", "-rs")
    parser.add_argument("--epsilon", "-ep")
    parser.add_argument("--horizon", "-hz")
    parser.add_argument("--allowBreaks", "-br")
    args = parser.parse_args()

    banditInstance  = parseArg(args.instance, "in", "--instance should be a path to the instance file")
    algorithm       = parseArg(args.algorithm, "al", "--algorithm should be one of epsilon-greedy, ucb, kl-ucb, thompson-sampling, and thompson-sampling-with-hint")
    randomSeed      = parseArg(args.randomSeed, "rs", "--randomSeed should be a non-negative integer")
    epsilon         = parseArg(args.epsilon, "ep", "--epsilon should be a number in [0, 1]")
    horizon         = parseArg(args.horizon, "hz", "--horizon should be a non-negative integer")
    allowBreaks     = (args.allowBreaks == "a")

    return

"""
    Run epsilon Greedy algorithm
    Assumptions:
        No first few pulls
        how ties get broken
            When multiple arms have same empirical means then first arm is picked
        algorithm-specific parameters
            epsilon provided in the command line arguments
"""
def runEG():
    global banditInstance, epsilon, horizon, REG

    REG = 0
    for t in range(horizon):
        if (allowBreaks and (t in breakPoints)):
            printOutput(numIters=t)
        
        if (np.random.binomial(1, epsilon) == 1):
            REG += banditInstance.highestMeanReward - banditInstance.pullArm()
        else:
            armEmpiricalMeans = np.asarray(banditInstance.getEmpiricalMeans())
            REG += banditInstance.highestMeanReward - banditInstance.pullArm(armEmpiricalMeans.argmax())
    
    return

"""
    Run UCB algorithm
    Assumptions:
        first few pulls
            1 Round of Round Robin
        how ties get broken
            When multiple arms have same ucbs then first arm is picked
        no algorithm-specific parameters
"""
def runUCB():
    global banditInstance, horizon, REG

    REG = 0
    for t in range(horizon):
        if (allowBreaks and (t in breakPoints)):
            printOutput(numIters=t)
        
        if (t < banditInstance.numArms):
            REG += banditInstance.highestMeanReward - banditInstance.pullArm(t)
        else:
            armUCBs = np.asarray(banditInstance.getUCBs(t))
            REG += banditInstance.highestMeanReward - banditInstance.pullArm(armUCBs.argmax())
    
    return

"""
    Run KL-UCB algorithm
    Assumptions:
        first few pulls
            1 Round of Round Robin
        how ties get broken
            When multiple arms have same kl-ucbs then first arm is picked
        algorithm-specific parameters
            c is 3
            max 50 rounds and precision 0.001 for searching appropiate kl-ucb
"""
def runKLUCB():
    global banditInstance, horizon, REG

    REG = 0
    for t in range(horizon):
        if (allowBreaks and (t in breakPoints)):
            printOutput(numIters=t)
 
        if (t < banditInstance.numArms):
            REG += banditInstance.highestMeanReward - banditInstance.pullArm(t)
        else:
            armKLUCBs = np.asarray(banditInstance.getKLUCBs(t, 3))
            REG += banditInstance.highestMeanReward - banditInstance.pullArm(armKLUCBs.argmax())
    
    return

"""
    Run Thompson Sampling algorithm
    Assumptions:
        no first few pulls
        how ties get broken
            When multiple arms have same thompsonSamples then first arm is picked
        no algorithm-specific parameters
"""
def runTS():
    global banditInstance, horizon, REG, allowBreaks, breakPoints

    REG = 0
    for t in range(horizon):
        if (allowBreaks and (t in breakPoints)):
            printOutput(numIters=t)

        thompsonSamples = np.asarray(banditInstance.getThompsonSamples())
        REG += banditInstance.highestMeanReward - banditInstance.pullArm(thompsonSamples.argmax())
    
    return

"""
    Run Thompson Sampling with Hint algorithm
    Assumptions:
        first few pulls
            Thompson Sampling in exploration phase. Exploration phase has max(0.016*horizon, 2*numArms) rounds
        how ties get broken
            When multiple arms have same thompsonSamples or empricalMeans then first arm is picked
        algorithm-specific parameters
            0.016 as epsilon for exploration phase
            0.01 as precision for choosing in exploitation phase
"""
def runTSWH():
    global banditInstance, horizon, REG, allowBreaks, breakPoints

    precision = 0.01
    numExploration = 0.016 * horizon
    if (numExploration < 2 * banditInstance.numArms):
        numExploration = 2 * banditInstance.numArms
    REG = 0
    for t in range(horizon):
        if (allowBreaks and (t in breakPoints)):
            printOutput(numIters=t)

        if (t < numExploration):
            thompsonSamples = np.asarray(banditInstance.getThompsonSamples())
            REG += banditInstance.highestMeanReward - banditInstance.pullArm(thompsonSamples.argmax())
        else:
            empiricalMeans = np.asarray(banditInstance.getEmpiricalMeans())
            meanDifferences = np.absolute(empiricalMeans - banditInstance.highestMeanReward)
            if (np.min(meanDifferences) < precision):
                REG += banditInstance.highestMeanReward - banditInstance.pullArm(meanDifferences.argmin())
            else:
                thompsonSamples = np.asarray(banditInstance.getThompsonSamples())
                REG += banditInstance.highestMeanReward - banditInstance.pullArm(thompsonSamples.argmax())
            
    return

if __name__ == "__main__":
    # Reading inputs
    parseArgs()

    # Setting randomSeed
    np.random.seed(randomSeed)

    if (algorithm == AL_EG): runEG()
    elif (algorithm == AL_UCB): runUCB()
    elif (algorithm == AL_KLUCB): runKLUCB()
    elif (algorithm == AL_TS): runTS()
    elif (algorithm == AL_TSWH): runTSWH()

    # print(banditInstance)
    printOutput()
