import numpy as np

""" Get direction from action """
def getDirection(a):
    """
        Actions
            Up      --  North   --  0
            Down    --  South   --  1
            Right   --  East    --  2
            Left    --  West    --  3
    """
    try:
        if (int(a) == 0): return "N"
        elif (int(a) == 1): return "S"
        elif (int(a) == 2): return "E"
        elif (int(a) == 3): return "W"
        else: raise Exception("Invalid Action")
    except Exception as err:
        print(err)
        exit()

""" Get state number from (i, j) in grid """
def getState(i, j):
    return i * nCols + j

""" Returns start state from index of grid """
def getStartState(startIndex):
    if len(startIndex) != 1:
        print("Multiple start indices, invalid input")
        exit()

    return getState(startIndex[0][0], startIndex[0][1])

""" Returns end states from index of grid """
def getEndStates(endIndices):
    states = []
    for ind in endIndices:
        states.append(getState(ind[0], ind[1]))

    return states

""" Returns new state based on action """
def updateState(currentState, action):
    try:
        if (int(action) == 0): return currentState - nCols
        elif (int(action) == 1): return currentState + nCols
        elif (int(action) == 2): return currentState + 1
        elif (int(action) == 3): return currentState - 1
        else: raise Exception("Invalid Action")
    except Exception as err:
        print(err)
        exit()

""" Parses input arguments and returns args """
def parseArgs():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", "-g", required=True)
    parser.add_argument("--value_policy", "-vp", required=True)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # Reading grid path
    args = parseArgs()
    grid_path = args.grid
    value_policy_path = args.value_policy

    grid, value_policy = None, None
    value_policy
    try:
        with open(grid_path, 'r') as f:
            grid = np.array([line.strip().split() for line in f], dtype=int)
        with open(value_policy_path, 'r') as f:
            value_policy = np.array([line.strip().split() for line in f], dtype=float)
    except Exception as err:
        print(err)
        exit()

    nCols   = len(grid[0])
    nRows   = len(grid)

    endStates   = getEndStates(np.argwhere(grid == 3))
    currentState= getStartState(np.argwhere(grid == 2))

    path = ""
    while currentState not in endStates:
        _, action = value_policy[currentState]
        path += getDirection(action) + " "
        currentState = updateState(currentState, action)
    
    print(path.strip())
