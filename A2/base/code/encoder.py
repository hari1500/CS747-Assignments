import numpy as np

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
    retStr = ""
    for ind in endIndices:
        retStr += str(getState(ind[0], ind[1])) + " "

    return retStr.strip()

""" prints All Transitions """
def printTransitions():
    for i in range(nRows):
        for j in range(nCols):
            if(grid[i][j] != 1):
                if(i > 0):
                    if (grid[i-1][j] == 3):
                        print("transition "+str(getState(i, j))+" "+str(0)+" "+str(getState(i-1, j))+" 10000000 1")
                    elif (grid[i-1][j] != 1):
                        print("transition "+str(getState(i, j))+" "+str(0)+" "+str(getState(i-1, j))+" -1 1")
                if(j < nCols-1):
                    if (grid[i][j+1] == 3):
                        print("transition "+str(getState(i, j))+" "+str(2)+" "+str(getState(i, j+1))+" 10000000 1")
                    elif (grid[i][j+1] != 1):
                        print("transition "+str(getState(i, j))+" "+str(2)+" "+str(getState(i, j+1))+" -1 1")
                if(i < nRows-1):
                    if (grid[i+1][j] == 3):
                        print("transition "+str(getState(i, j))+" "+str(1)+" "+str(getState(i+1, j))+" 10000000 1")
                    elif (grid[i+1][j] != 1):
                        print("transition "+str(getState(i, j))+" "+str(1)+" "+str(getState(i+1, j))+" -1 1")
                if(j > 0):
                    if (grid[i][j-1] == 3):
                        print("transition "+str(getState(i, j))+" "+str(3)+" "+str(getState(i, j-1))+" 10000000 1")
                    elif (grid[i][j-1] != 1):
                        print("transition "+str(getState(i, j))+" "+str(3)+" "+str(getState(i, j-1))+" -1 1")

""" Parses input arguments and returns grid file path """
def parseArgs():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", "-g", required=True)
    args = parser.parse_args()

    return args.grid

if __name__ == "__main__":
    # Reading grid path
    grid_path = parseArgs()

    grid = None
    try:
        with open(grid_path, 'r') as f:
            grid = np.array([line.strip().split() for line in f], dtype=int)
    except Exception as err:
        print(err)
        exit()

    nCols   = len(grid[0])
    nRows   = len(grid)

    """
        Actions
            Up      --  North   --  0
            Down    --  South   --  1
            Right   --  East    --  2
            Left    --  West    --  3
    """
    nActions    = 4
    nStates     = nCols * nRows
    discount    = 0.9
    mdpType     = "episodic"
    startIndex  = np.argwhere(grid == 2)
    endIndices  = np.argwhere(grid == 3)

    print("numStates %s" % (nStates))
    print("numActions %s" % (nActions))
    print("start %s" % getStartState(startIndex))
    print("end %s" % getEndStates(endIndices))
    printTransitions()
    print("mdptype %s" % (mdpType))
    print("discount %s" % (discount))