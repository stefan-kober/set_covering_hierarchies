import numpy as np
from scipy import sparse
from gurobipy import *
from hierarchies import *
from sortFunctions import *
from examples import *
from workflows import *
from separation import *

#example for iterated separation
n=10
m=50
d=3
g=generateRandomSCProblem(n, m, d)
obj=[round(np.random.rand(),3) for i in range(g.shape[1])]
strengthenRelaxation(g,2000,obj,hierarchy=fhw)
strengthenRelaxation(g,2000,obj,hierarchy=bz)

#example for one application of hierarchies
compareTimeAndResultSC(50,400,3,10)
