import numpy as np
from gurobipy import *
from scipy import sparse
from hierarchies import *
from sortFunctions import *

'''Separate for extended formulations: given a solution to a relaxed polytope, check if this solution is in the extended formulation and if not, generate an inequality in the original space, valid for the extended formulation, separating the solution from the extended formulation.
Parameters: 
    - xstar: solution to the relaxed problem
    - extendedFormulation: extendedFormulation that needs to be checked
Output: 
    - m.ObjVal: objectiveValue of the separation problem. If it is less than 0, an inequality was found.
    - m.Runtime: time gurobipy needed to solve the separation problem
        - res: incidence vector of the resulting inequality
        - res_ext-res_ext: rhs of the resulting inequality'''
def separate(xstar,extendedFormulation):
    num_ps=int(sum(extendedFormulation.proj.toarray())[0])#number of polytopes that were united to obtain the extended formulation
    num_dims=int((extendedFormulation.ineqs.shape[1]-2)/num_ps)#number of dimensions of the original problem
    X=np.array(extendedFormulation.ineqs.toarray())
    A=[X[:,i*num_dims:(i+1)*num_dims].transpose() for i in range(num_ps)]
    B=X[:,-num_ps:].transpose()
    d=np.negative(np.array([x[0] for x in extendedFormulation.rhs.toarray()]))
    m=Model("separation")
    m.setParam('OutputFlag', False)
    obj=d
    for i in range(num_dims):
        obj+=xstar[i]*A[0][i][:]
    x = m.addVars(range(extendedFormulation.ineqs.shape[0]),vtype=GRB.CONTINUOUS,obj=obj,name="x")
    m.update()
    for i in B:
        expr = 0
        for index,v in [(index,v) for index,v in enumerate(i) if v!=0]:
            expr += v*x[index]
        m.addConstr(expr == 0)
    
    a_expr=[]
    for i in A[0]:
        a_expr_i=0
        for index,v in [(index,v) for index,v in enumerate(i) if v!=0]:
            a_expr_i += v*x[index]
        a_expr.append(a_expr_i)
    
    for a in A[1:]:
        for index,i in enumerate(a):
            expr=0
            for index2,v in [(index2,v) for index2,v in enumerate(i) if v!=0]:
                expr -= v*x[index2]
            m.addConstr(expr + a_expr[index] == 0)
    
    expr=0    
    for index,i in enumerate(X):
        expr+=x[index]
    m.addConstr(expr==1)
    
    m.update()
    m.ModelSense=1
    
    #m.write("model%s.lp"%step)
    
    m.optimize()
    x=m.getVars()
    if m.Status==GRB.OPTIMAL:
        print("Opt. Value=%s"%m.ObjVal)
        res_ext=[]
        for i in range(len(x)):
            res_ext.append(x[i].X)
        res = [np.dot(res_ext,i) for i in extendedFormulation.ineqs.transpose().toarray()]
        print(str(res[:num_dims])+">="+str(res_ext[-2]-res_ext[-1]))
        return (m.ObjVal,round(m.Runtime,4),(res[:num_dims],res_ext[-2]-res_ext[-1]))
    return    

'''Method to strengthen the relaxation of a given problem using a given hierarchy, adding one inequality at a time by separation for a given number of steps. Output to file and to standard output
Parameters:
    - g: the problem
    - steps: the number of steps
    - obj: the objective vector
    - hierarchy: the hierarchy used to strengthen the relaxation
    - randomness: randomness when choosing the extended formulation'''
def strengthenRelaxation(g, steps, obj, hierarchy, randomness=0.2):
    script_dir = os.path.dirname(__file__)
    file = open(os.path.join(script_dir, "../output/separation_%s_n%sm%s.txt"%(hierarchy.__name__,g.shape[1],g.shape[0])), "w+")
    file.write("dims: "+str(g.shape[1])+", ineqs: "+str(g.shape[0])+"\r")
    counter=0
    opt=False
    ineqs=g.toarray()
    g_problem=sc_problem(g,obj)
    optv=g_problem.solveDirectly()[0]
    model_relaxed=g_problem.createModel()
    c=hierarchy(g_problem,debug=True)
    for i in range(steps):
        print("step %s"%i)
        sol_relaxed=g_problem.solveModel(model_relaxed)
        file.write("(%s,%s)\r"%(i,(round(optv/sol_relaxed[0],3))))
        if all(round(e,4)==0 or round(e,4)==1 for e in sol_relaxed[2]):
            opt=True
            break
        xstar=sol_relaxed[2]
        if hierarchy==bfn:
            b=sepn_lowestDistFromHalf(c, xstar, ineqs, randomness)
        else:
            b=sepe_noOne(c, xstar, ineqs, randomness)
        sep=separate(sol_relaxed[2],b[0])
        new_ineq=sep[2]
        if all(round(e,4)==0 for e in new_ineq[0]):
            counter+=1
        else:
            for ef in c:
                ef.addConstraint(new_ineq[0],new_ineq[1])
            expr=0
            x=model_relaxed.getVars()
            for index,v in enumerate(new_ineq[0]):
                expr+=v*x[index]
            model_relaxed.addConstr(expr >= new_ineq[1])
    if not opt:
        print("step %s"%steps)
        g_problem.solveModel(model_relaxed)
    g_problem.solveDirectly()
    print("unnecessary steps: %s"%counter)