'''In this file, I will implement the boolean formula method for a hierarchy for set-covering problems, as described by FIORINI, HUYNH and WELTGE, as well as the algorithm designed by BIENSTOCK and ZUCKERBERG.'''

import numpy as np
from gurobipy import *
from extendedFormulation import *
from sortFunctions import *
from copy import *
from scipy import sparse

'''The sc_problem class models a set covering problem. It holds all of the inequalities of the set covering problem, as well as an approximation of the polytope (cur). This approximation is higher-dimensional and comes together with a projection back into the normal space.
Parameters:
    - ineqs: 0-1 matrix with n columns and m rows representing the constraints of the model
    - obj: the objective vector of the set covering problem. The default value is the ones-vector.'''
class sc_problem():
    def __init__(self, ineqs, obj=0):
        self.num_ineqs=ineqs.shape[0]#number of inequalities of the original problem
        self.num_dims=ineqs.shape[1]#number of dimensions of the original problem
        self.orig_ineqs=ineqs
        self.ineqs=ineqs#inequalities of original problem
        self.ineqs=sparse.construct.vstack([self.ineqs,sparse.construct.identity(self.num_dims),-sparse.construct.identity(self.num_dims)])
        self.rhs=sparse.construct.hstack([sparse.coo_matrix(np.ones(self.num_ineqs)),sparse.coo_matrix(np.zeros(self.num_dims)),-sparse.coo_matrix(np.ones(self.num_dims))])#right hand side of inequalities
        self.approximation=extendedFormulation(self.ineqs.copy(),self.rhs.copy(),sparse.construct.identity(self.num_dims))#current state of the approximation
        if obj==0:#objective for the problem; default is min(sum(x_i))
            self.obj=np.ones(self.num_dims)
        else:
            self.obj=obj

    '''Create a gurobipy model to solve the setcovering problem with respect to the current approximation.
    Parameters:
        - self: the polytope and the objective vector
    Output:
        - createModel(): the gurobipy model'''
    def createModel(self):
        return self.approximation.createModel(self.obj)
    
    '''Solve a gurobipy (mixed integer) linear program associated with the approximation.
    Parameters:
        - self: the approximation
    Output:
        - solveModel(): the result of the call to solving the model in the approximation'''
    def solveModel(self,m):
        return self.approximation.solveModel(m)
    
    '''Create a gurobipy model to solve the setcovering problem with respect to the current approximation, with only binary variables, and solve the model.
    Parameters:
        - self: the approximation
    Output:
        - solveModel(): the result of the call to solving the model in the approximation'''
    def solveDirectly(self):
        return self.approximation.solveModel(self.approximation.createModel(self.obj,binary=True))
    
    '''Compare the different hierarchies on the given model.
    Parameters:
        - self: the underlying model
        - sortFuns: different sort functions for the bz-algorithm
    Output:
        - optValues: vector of the results'''
    def compare(self, sortFuns):
        model=self.createModel()
        optValues=[self.solveModel(model)]
        solFHW=fhw(self)
        solBZ=[bz(self,fun) for fun in sortFuns]
        model=solFHW.createModel(self.obj)
        optValues.append(solFHW.solveModel(model))
        optValues+=[s.solveModel(s.createModel(self.obj)) for s in solBZ]
        optValues+=[self.solveDirectly()]
        return optValues
        
'''The function fhw applies one round of the boolean formula method to the given input. The required input is a set-covering problem (defined by a matrix which represents its inequalities) and the current state of the problem (typically either the 0-1-cube, the relaxed problem or a further step). This again is defined by a matrix together with a vector, representing the inequalities, as well as another matrix, representing the respective projection. The underlying algorithm was discovered by FIORINI, HUYNH and WELTGE.
Parameters:
    - problem: the setcovering problem
    - debug: if debug is set to True, the output is a list of extended formulations corresponding to the edges, instead of their intersection
Output:
    - unions_list: the result of the application of the hierarchy to the approximation with respect to the output format specified by the debug parameter'''
def fhw(problem, debug=False):
    '''In the first step, the approximation is intersected with all the hyperplanes of the form x_i=1. Therefore the hyperplanes need to be shifted into the higherdimensional space. The result is stored as a seperate list of arrays.'''
    intersections = []
    intersection_rhs=sparse.construct.hstack([problem.approximation.rhs,sparse.coo_matrix(1)])
    for i in range(problem.num_dims):
        hyperplane=np.zeros(problem.num_dims)
        hyperplane[i]=1
        hyperplane=sparse.coo_matrix(hyperplane)
        hyperplane=problem.approximation.proj.dot(hyperplane.transpose())
        intersection=sparse.construct.vstack([problem.approximation.ineqs.copy(),hyperplane.transpose()])#add hyperplane to inequalities of the approximation
        intersections.append(extendedFormulation(intersection,intersection_rhs,problem.approximation.proj))
    '''Each intersection can be identified with a corresponding dimension. In the next step, we look at all of the inequalities of the original problem on their own: we take the convex hull of the union of all the intersections where the corresponding variable appears in the inequality. Therefore, the algorithm by Balas for the union of polytopes is used.'''
    unions_list=[]
    for ineq in problem.orig_ineqs.toarray()[:]:
        subpolytopes=[]
        for index,i in enumerate(ineq):
            if i==1:
                subpolytopes.append(intersections[index])
        unions_list.append(balas_list(subpolytopes))
    '''In the last step, all of the resulting unions need to be intersected to achieve the end result. As all the unions are extended formulations, the trivial implementation of this causes a blowup of the dimension. The current implementation allows to intersect two extended formulations of arbitrary dimension that project into the same space.'''
    if debug:
        return unions_list
    return intersect_list(unions_list)

'''The function bz applies one round of the algorithm designed by BIENSTOCK and ZUCKERBURG to the given input. The required input is a set-covering problem (defined by a matrix which represents its inequalities) and the current state of the problem (typically either the 0-1-cube, the relaxed problem or a further step). This again is defined by a matrix together with a vector, representing the inequalities, as well as another matrix, representing the respective projection.
Parameters:
    - problem: the setcovering problem
    - debug: if debug is set to True, the output is a list of extended formulations corresponding to the edges, instead of their intersection
Output:
    - unions_list: the result of the application of the hierarchy to the approximation with respect to the output format specified by the debug parameter'''
def bz(problem, sortFun=bz_randomPermutation, debug=False):
    '''In the first step, the approximation is intersected with all the hyperplanes of the form x_i=1. Therefore the hyperplanes need to be shifted into the higherdimensional space. The result is stored as a seperate list of arrays.'''
    intersections = []
    intersection_rhs=sparse.construct.hstack([problem.approximation.rhs,sparse.coo_matrix(1)])
    for i in range(problem.num_dims):
        hyperplane=np.zeros(problem.num_dims)
        hyperplane[i]=1
        hyperplane=sparse.coo_matrix(hyperplane)
        hyperplane=problem.approximation.proj.dot(hyperplane.transpose())
        intersection=sparse.construct.vstack([problem.approximation.ineqs.copy(),hyperplane.transpose()])#add hyperplane to inequalities of the approximation
        intersections.append(extendedFormulation(intersection,intersection_rhs,problem.approximation.proj))
    '''Each intersection can be identified with a corresponding dimension. In the next step, we look at all of the inequalities of the original problem on their own: first we restrict the projections of which the corresponding variable appears in the inequality according to the previous ordering. In the following we calculate the convex hull of the union of the restrictions. Therefore, the algorithm by Balas for the union of polytopes is used.'''
    unions_list=[]
    for ineq in problem.orig_ineqs.toarray()[:]:
        subpolytopes=[]
        for index,i in enumerate(ineq):
            if i==1:
                subpolytopes.append((intersections[index],index))
        subpolytopes=sortFun(problem,subpolytopes)
        restrictions=restrictForBZ(subpolytopes, problem.num_dims)
        unions_list.append(balas_list([x[0] for x in restrictions]))
    '''In the last step, all of the resulting unions need to be intersected to achieve the end result. As all the unions are extended formulations, the trivial implementation of this causes a blowup of the dimension. The current implementation allows to intersect two extended formulations of arbitrary dimension that project into the same space.'''
    if debug:
        return unions_list
    return intersect_list(unions_list)

'''An experimental hierarchy performing brute force on the nodes. Inspired by LOVASZ-SCHRIJVER
Parameters:
    - problem: the setcovering problem
    - debug: if debug is set to True, the output is a list of extended formulations corresponding to the edges, instead of their intersection
Output:
    - unions_list: the result of the application of the hierarchy to the approximation with respect to the output format specified by the debug parameter'''
def bfn(problem, debug=False):
    '''In the first step, the approximation is intersected with all the hyperplanes of the form x_i=0 and x_i=1. Therefore the hyperplanes need to be shifted into the higherdimensional space. The result is stored as a seperate list of arrays.'''
    intersections0 = []
    intersections1 = []
    intersection0_rhs=sparse.construct.hstack([problem.approximation.rhs,sparse.coo_matrix(0)])
    intersection1_rhs=sparse.construct.hstack([problem.approximation.rhs,sparse.coo_matrix(1)])
    for i in range(problem.num_dims):
        hyperplane0=np.zeros(problem.num_dims)
        hyperplane0[i]=-1
        hyperplane0=sparse.coo_matrix(hyperplane0)
        hyperplane0=problem.approximation.proj.dot(hyperplane0.transpose())
        hyperplane1=np.zeros(problem.num_dims)
        hyperplane1[i]=1
        hyperplane1=sparse.coo_matrix(hyperplane1)
        hyperplane1=problem.approximation.proj.dot(hyperplane1.transpose())
        intersection0=sparse.construct.vstack([problem.approximation.ineqs.copy(),hyperplane0.transpose()])#add hyperplane to inequalities of the approximation
        intersection1=sparse.construct.vstack([problem.approximation.ineqs.copy(),hyperplane1.transpose()])#add hyperplane to inequalities of the approximation
        intersections0.append(extendedFormulation(intersection0,intersection0_rhs,problem.approximation.proj))
        intersections1.append(extendedFormulation(intersection1,intersection1_rhs,problem.approximation.proj))
    '''Each intersection can be identified with a corresponding dimension. In the next step, we look at all of the variables of the original problem: for each variable, we take the convex hull of the union of the projection to zero and one. Therefore, the algorithm by Balas for the union of polytopes is used.'''
    unions_list=[]
    for d in range(problem.num_dims):
        unions_list.append(balas_list([intersections1[d],intersections0[d]]))
    '''In the last step, all of the resulting unions need to be intersected to achieve the end result. As all the unions are extended formulations, the trivial implementation of this causes a blowup of the dimension. The current implementation allows to intersect two extended formulations of arbitrary dimension that project into the same space.'''
    if debug:
        return unions_list
    return intersect_list(unions_list)

'''Another experimental hierarchy, that only works on graphs. It works by trying to bruteforce solutions for each edge: for an edge, the nodes are set to (0,1),(1,0) and (1,1) respectively in order to try all possible configurations. This hierarchy needs considerably less steps than the other ones when working on graphs.
Parameters:
    - problem: the setcovering problem
    - debug: if debug is set to True, the output is a list of extended formulations corresponding to the edges, instead of their intersection
Output:
    - unions_list: the result of the application of the hierarchy to the approximation with respect to the output format specified by the debug parameter'''
def bfe(problem, debug=False):
    unions_list=[]
    for ineq in problem.orig_ineqs.toarray()[:]:
        subpolytopes=[]
        dims=[]
        for index,i in enumerate(ineq):
            if i==1:
                dims.append(index)
        subpolytopes.append(intersect_hyperplane(intersect_hyperplane(problem.approximation,dims[0],1),dims[1],0))
        subpolytopes.append(intersect_hyperplane(intersect_hyperplane(problem.approximation,dims[0],1),dims[1],1))
        subpolytopes.append(intersect_hyperplane(intersect_hyperplane(problem.approximation,dims[0],0),dims[1],1))
        unions_list.append(balas_list(subpolytopes))
    if(debug):
        return unions_list
    return intersect_list(unions_list)

'''The following helper function restricts the projections according to the ordering used in the bz-algorithm. Therefore every subpolytope in the list is intersected with the hyperplane x_i=0 if there is a projection before it in the list of subpolytopes that is projected to the hyperplane x_i=1.
Parameters:
    - subpolytopes: list of extended formulations to be restricted
    - dim: number of dimensions of the original problem
Output:
    - restrictions: list of the required restrictions for bz'''
def restrictForBZ(subpolytopes,dim):
    restrictions = [(deepcopy(i[0]),i[1]) for i in subpolytopes]
    for index,s in enumerate(restrictions):
        if index<len(restrictions)-1:
            rest_vec=np.zeros(dim)
            rest_vec[s[1]]=-1
            rest_vec=sparse.coo_matrix(rest_vec)
            for s2 in restrictions[index+1:]:
                rest_ineq=rest_vec.dot(s2[0].proj)
                s2[0].ineqs=sparse.construct.vstack([s2[0].ineqs,rest_ineq])
                s2[0].rhs=sparse.construct.hstack([s2[0].rhs,0])
    return restrictions
    
'''Apply the compare-function to a large number of problems.
Parameters:
    - ineqSystems: list containing the constraints for the problems
    - sortFuns: sortFunctions that should be used for bz
Output:
    - differences: return the constraints of the problems, where some sort-algorithms for bz perform better than fhw
'''
def bulkCompare(ineqSystems, sortFuns):
    probs = [sc_problem(i) for i in ineqSystems]
    res = [p.compare(sortFuns) for p in probs]
    differences=[]
    for index,r in enumerate(res):
        if not all(round(x-r[1],3)==0 for x in r[2:]):
            differences.append(ineqSystems[index])
    return differences