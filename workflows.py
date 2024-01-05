'''In this file, several complex workflows are collected'''

import numpy as np
from gurobipy import *
from hierarchies import *
from sortFunctions import *
from examples import *
import os

'''Compare the time and the result of the relaxed solution vs the solution strengthened with one entire application of the hierarchies vs the optimal solution on a vertex covering problem with n dimensions and m constraints.
Parameters:
    - n: number of dimensions
    - m: maximum number of constraints
    - amount: number of problems'''
def compareTimeAndResultVC(n,m,amount):
    script_dir = os.path.dirname(__file__)
    
    file = open(os.path.join(script_dir, "../output/n%sm%samount%s.txt"%(n,m,amount)), "w+")
    file.write("dims: "+str(n)+", ineqs: "+str(m)+", amount: "+str(amount)+"\r")
    IG_rel=[]
    IG_bfn=[]
    IG_bfe=[]
    
    for i in range(amount):
        g=generateRandomGraph(n,m)
        a=sc_problem(g,[round(np.random.rand(),3) for j in range(n)])
        
        print("optimal solution")
        sol_f=a.solveDirectly()
        print(sol_f)
        file.write("\r\topt:\r\t\toptVal: %s\r\t\ttime: %s"%(str(sol_f[0]),str(sol_f[1])))
        optVal=sol_f[0]
        
        print("relaxed solution")
        sol_a=a.solveModel(a.approximation.createModel(a.obj))
        IG=optVal/sol_a[0]
        file.write("\r\trelaxed:\r\t\toptVal: %s\r\t\ttime: %s"%(str(sol_a[0]),str(sol_a[1])))
        file.write("\r\t\tIG: %s"%str(IG))
        IG_rel.append(IG)
        
        b=bfn(a)
        print("bfn done")
        sol_b=b.solveModel(b.createModel(a.obj))
        IG=optVal/sol_b[0]
        file.write("\r\tbfn:\r\t\toptVal: %s\r\t\ttime: %s"%(str(sol_b[0]),str(sol_b[1])))
        file.write("\r\t\tIG: %s"%str(IG))
        b=0
        IG_bfn.append(IG)
        
        e=bfe(a)
        print("bfe done")
        sol_e=e.solveModel(e.createModel(a.obj))
        IG=optVal/sol_e[0]
        file.write("\r\tbfe:\r\t\toptVal: %s\r\t\ttime: %s"%(str(sol_e[0]),str(sol_e[1])))
        file.write("\r\t\tIG: %s"%str(IG))
        e=0
        IG_bfe.append(IG)
        
        file.flush()
    
    file.write("\r\rIG_rel=%s"%round(sum(IG_rel)/amount,4))
    file.write("\rIG_bfn=%s"%round(sum(IG_bfn)/amount,4))
    file.write("\rIG_bfe=%s"%round(sum(IG_bfe)/amount,4))
    file.close()
    
'''Compare the time and the result of the relaxed solution vs the solution strengthened with one entire application of the hierarchies vs the optimal solution on a set covering problem with n dimensions and m constraints.
Parameters:
    - n: number of dimensions
    - m: maximum number of constraints
    - d: cardinality of a hyperedge
    - amount: number of problems'''
def compareTimeAndResultSC(n,m,d,amount):
    script_dir = os.path.dirname(__file__)
    
    file = open(os.path.join(script_dir, "../output/n%sm%sd%samount%s.txt"%(n,m,d,amount)), "w+")
    file.write("dims: "+str(n)+", ineqs: "+str(m)+", cardinality: "+str(d)+", amount: "+str(amount)+"\r")
    IG_rel=[]
    IG_bz=[]
    IG_fhw=[]
    
    for i in range(amount):
        g=generateRandomSCProblem(n, m, d)
        a=sc_problem(g,[round(np.random.rand(),3) for j in range(n)])
        
        print("optimal solution")
        sol_f=a.solveDirectly()
        print(sol_f)
        file.write("\r\topt:\r\t\toptVal: %s\r\t\ttime: %s"%(str(sol_f[0]),str(sol_f[1])))
        optVal=sol_f[0]
        
        print("relaxed solution")
        sol_a=a.solveModel(a.approximation.createModel(a.obj))
        IG=optVal/sol_a[0]
        file.write("\r\trelaxed:\r\t\toptVal: %s\r\t\ttime: %s"%(str(sol_a[0]),str(sol_a[1])))
        file.write("\r\t\tIG: %s"%str(IG))
        IG_rel.append(IG)
        
        b=fhw(a)
        print("fhw done")
        sol_b=b.solveModel(b.createModel(a.obj))
        IG=optVal/sol_b[0]
        file.write("\r\tfhw:\r\t\toptVal: %s\r\t\ttime: %s"%(str(sol_b[0]),str(sol_b[1])))
        file.write("\r\t\tIG: %s"%str(IG))
        b=0
        IG_fhw.append(IG)
        
        e=bz(a)
        print("bz done")
        sol_e=e.solveModel(e.createModel(a.obj))
        IG=optVal/sol_e[0]
        file.write("\r\tbz:\r\t\toptVal: %s\r\t\ttime: %s"%(str(sol_e[0]),str(sol_e[1])))
        file.write("\r\t\tIG: %s"%str(IG))
        e=0
        IG_bz.append(IG)
        
        file.flush()
    
    file.write("\r\rIG_rel=%s"%round(sum(IG_rel)/amount,4))
    file.write("\rIG_fhw=%s"%round(sum(IG_fhw)/amount,4))
    file.write("\rIG_bz=%s"%round(sum(IG_bz)/amount,4))
    file.close()