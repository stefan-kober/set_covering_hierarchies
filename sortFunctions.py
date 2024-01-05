''''In this file, several sorting functions are implemented, which are applicable for the bz hierarchy or the separation process.'''

import numpy as np

def bz_depSortReverse(problem, subpolytopes):
    dependencies=[sum(d) for d in problem.orig_ineqs.transpose().toarray()]
    return sorted(subpolytopes,key=lambda x:dependencies[x[1]], reverse=True)

def bz_depSort(problem, subpolytopes):
    dependencies=[sum(d) for d in problem.orig_ineqs.transpose().toarray()]
    return sorted(subpolytopes,key=lambda x:dependencies[x[1]])

def bz_randomPermutation(problem, subpolytopes):
    return sorted(subpolytopes,key=lambda x:np.random.rand())

def sepn_lowestDistFromHalf(efs,xstar,ineqs,randomness):
    return [y[1] for y in sorted(enumerate(efs), key=lambda x:abs(0.5-xstar[x[0]]))]

def sepe_lowestEdgeSum(efs,xstar,ineqs,randomness):
    return [y[1] for y in sorted(enumerate(efs), key=lambda x:np.dot(ineqs[x[0],:],xstar)+2*randomness*np.random.rand())]

def sep_random(efs,xstar,ineqs,randomness):
    return sorted(efs, key=lambda x:np.random.rand())

def sepn_weightedDistFromHalf(efs,xstar,ineqs,randomness):
    strength=sum(ineqs.transpose())
    return [y[1] for y in sorted(enumerate(efs), key=lambda x:(abs(0.5-xstar[x[0]])+randomness*np.random.rand())/strength[x[0]])]

def sepe_lowestDistFromHalf(efs,xstar,ineqs,randomness):  
    return [y[1] for y in sorted(enumerate(efs), key=lambda x:np.dot(ineqs[x[0],:],abs(0.5-np.array(xstar)))+randomness*np.random.rand())]

def sepe_lowestDistFromWeakestPoint(efs,xstar,ineqs,randomness):  
    return [y[1] for y in sorted(enumerate(efs), key=lambda x:np.dot(ineqs[x[0],:],abs((1/sum(ineqs[x[0],:]))-np.array(xstar)))+randomness*np.random.rand())]

def sepe_noOne(efs,xstar,ineqs,randomness):
    return [y[1] for y in sorted(enumerate(efs), key=lambda x:(np.prod(1-ineqs[x[0],:]*xstar)),reverse=True)]
