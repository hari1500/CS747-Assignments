import numpy as np
import pulp as plp
from collections import defaultdict

mdp, algorithm = None, None

"""
    filePath    : string
    numStates   : integer
    numActions  : integer
    startState  : integer
    endStates   : [ integer ]
    mdpType     : string
    discount    : float
    transitions : { (fromState, action) --> (toState, reward, probability) }
"""
class MDP:
    def __init__(self, filePath, errStmt):
        try:
            self.filePath = filePath

            self.endStates = []
            self.transitions = defaultdict(list)
            with open(filePath, 'r') as f:
                lines = [line.strip() for line in f]
                for line in lines:
                    line = line.split()
                    if (line[0] == "numStates"):
                        self.numStates = int(line[1])
                    elif (line[0] == "numActions"):
                        self.numActions = int(line[1])
                    elif (line[0] == "start"):
                        self.startState = int(line[1])
                    elif (line[0] == "end"):
                        self.endStates = [int(s) for s in line[1:]]
                    elif (line[0] == "mdptype"):
                        self.mdpType = line[1]
                        if (line[1] not in ["continuing", "episodic"]):
                            raise Exception("mdptype should be either continuing or episodic")
                    elif (line[0] == "discount"):
                        self.discount = float(line[1])
                    elif (line[0] == "transition"):
                        self.transitions[(int(line[1]), int(line[2]))].append((int(line[3]), float(line[4]), float(line[5])))
        except FileNotFoundError:
            print(errStmt)
            exit()
        except Exception as err:
            print(err)
            exit()

    def __str__(self):
        return "MDP FILE Path: %s" % (self.filePath)

    def print(self):
        print("numStates", self.numStates)
        print("numActions", self.numActions)
        print("start", self.startState)
        print("end", self.endStates)
        print("mdptype", self.mdpType)
        print("discount", self.discount)
        print("transitions", self.transitions)

""" Parses input arguments """
def parseArgs():
    import argparse
    global mdp, algorithm
    def parseArg(arg, type, errStmt=""):
        try:
            if (type == "m"):
                return MDP(arg, errStmt)
            elif (type == "al"):
                if (arg in ["vi", "hpi", "lp"]): return arg

            print(errStmt)
            exit()
        except Exception:
            print(errStmt)
            exit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp", "-m", required=True)
    parser.add_argument("--algorithm", "-al", required=True)
    args = parser.parse_args()

    mdp         = parseArg(args.mdp, "m", "--mdp should be a path to input MDP file")
    algorithm   = parseArg(args.algorithm, "al", "--algorithm should be one of vi, hpi, lp")

    return

""" Print in final output format """
def printOutput(values, policies):
    for i in range(len(values)):
        print("%.6f %d" % (values[i], policies[i]))
    return

def runVI():
    global mdp
    values, delta = np.zeros(mdp.numStates), 1e-12

    def new_values_for_state(state, V_t):
        """ Returns new values for each action using V_t """
        new_values = np.zeros(mdp.numActions)
        for action in range(mdp.numActions):
            for (next_state, rew, prob) in mdp.transitions[(state, action)]:
                new_values[action] += prob * (rew + mdp.discount * V_t[next_state])
        return new_values
        # return np.array([ sum([ mdp.transitions[state][action][next_state][1] * (mdp.transitions[state][action][next_state][0] + mdp.discount * V_t[next_state]) for next_state in range(mdp.numStates)]) for action in range(mdp.numActions)])
    
    diff = delta
    while diff >= delta:
        diff = 0
        for s in range(mdp.numStates):
            if s in mdp.endStates:
                continue

            new_value = np.max(new_values_for_state(s, values))
            diff = max(diff, abs(new_value - values[s]))
            values[s] = new_value

    policies = [np.argmax(new_values_for_state(s, values)) for s in range(mdp.numStates)]

    printOutput(values, policies)
    return

def runHPI():
    global mdp
    values, policies, delta = np.zeros(mdp.numStates), np.zeros(mdp.numStates, dtype=int), 1e-4

    def new_values_for_state(state, V_t):
        """ Returns new values for each action using V_t """
        new_values = np.zeros(mdp.numActions)
        for action in range(mdp.numActions):
            for (next_state, rew, prob) in mdp.transitions[(state, action)]:
                new_values[action] += prob * (rew + mdp.discount * V_t[next_state])
        return new_values
    
    def values_as_per_policy(poli):
        """ Returns values as per policy """
        def get_lp_rhs(state, action, lp_vars):
            rhs = 0
            for (next_state, rew, prob) in mdp.transitions[(state, action)]:
                rhs += prob * (rew + mdp.discount * lp_vars[next_state])
            return rhs

        # lp_vars = [plp.LpVariable("V_%s" % (s), lowBound=0.0) for s in range(mdp.numStates)]
        lp_vars = plp.LpVariable.dicts("V" , list(range(mdp.numStates)), cat='Continuous')
        lp_prob = plp.LpProblem('Values')
        
        for s in range(mdp.numStates):
            if s in mdp.endStates:
                lp_prob += lp_vars[s] == 0
            else:
                lp_prob += lp_vars[s] == get_lp_rhs(s, poli[s], lp_vars)

        lp_prob.solve(plp.apis.PULP_CBC_CMD(msg=False))

        return [float(plp.value(lp_vars[s])) for s in range(mdp.numStates)]

    is_policy_optimal = False
    while not is_policy_optimal:
        values = values_as_per_policy(policies)
        new_policies = np.zeros(mdp.numStates)
        for s in range(mdp.numStates):
            if s in mdp.endStates:
                continue
            new_values = new_values_for_state(s, values)
            if (np.max(new_values) > values[s] + delta):
                new_policies[s] = np.argmax(new_values)
            else:
                new_policies[s] = policies[s]

        is_policy_optimal = np.all((new_policies - policies) == 0)
        policies = new_policies

    printOutput(values, policies)
    return

"""
    Reference for Pulp part: https://www.geeksforgeeks.org/python-linear-programming-in-pulp/
"""
def runLP():
    global mdp

    def new_values_for_state(state, V_t):
        """ Returns new values for each action using V_t """
        new_values = np.zeros(mdp.numActions)
        for action in range(mdp.numActions):
            for (next_state, rew, prob) in mdp.transitions[(state, action)]:
                new_values[action] += prob * (rew + mdp.discount * V_t[next_state])
        return new_values

    def get_lp_obj(vals):
        obj = 0
        for val in vals:
            obj += val
        return obj

    def get_lp_rhs(state, action, vals):
        rhs = 0
        for (new_state, rew, prob) in mdp.transitions[(state, action)]:
            rhs += prob * (rew + mdp.discount * vals[new_state])
        return rhs

    lp_vars = [plp.LpVariable("V_%s" % (s), lowBound=0.0) for s in range(mdp.numStates)]
    lp_prob = plp.LpProblem('Problem', plp.LpMinimize)
    lp_prob += get_lp_obj(lp_vars)
    
    for s in range(mdp.numStates):
        for a in range(mdp.numActions):
            lp_prob += lp_vars[s] >= get_lp_rhs(s, a, lp_vars)

    lp_prob.solve(plp.apis.PULP_CBC_CMD(msg=False))

    values = [plp.value(lp_vars[s]) for s in range(mdp.numStates)]
    policies = [np.argmax(new_values_for_state(s, values)) for s in range(mdp.numStates)]

    printOutput(values, policies)
    return


if __name__ == "__main__":
    # Reading inputs
    parseArgs()

    if(algorithm == "vi"): runVI()
    elif(algorithm == "hpi"): runHPI()
    elif(algorithm == "lp"): runLP()