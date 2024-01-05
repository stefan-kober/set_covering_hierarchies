import numpy as np
from gurobipy import *
from scipy import sparse

'''The extendedFormulation class models extended formulations of polytopes. The extended formulation of a polytope is essentially a higher-dimensional polytope together with a projection into the original space, that results in exactly the original polytope. Using extended formulations, we can calculate the intersections and unions of polytopes.
Parameters:
    - ineqs: 0-1 matrix, with n columns and m rows, representing the constraints on the model
    - rhs: vector with m entries, representing the right hand side of the inequalities
    - proj: 0-1 matrix, with n rows representing the projection of the extended formulation to the original space'''
class extendedFormulation():
    def __init__(self, ineqs, rhs, proj):
        self.ineqs=ineqs
        self.rhs=rhs
        self.proj=proj
    
    '''create a (mixed integer) linear programming model.
    Parameters:
        - self: the polytope
        - obj: the linear objective vector in the original space
        - binary: if set to true, all variables are binary
    Output:
        - m: the finished gurobipy model'''   
    def createModel(self, obj, binary=False):
        ext_obj=[np.dot(x,obj) for x in self.proj.toarray()]#lift the object to the space of the extended formulation
        m=Model("extendedFormulationLP")
        m.setParam( 'OutputFlag', False )#suppress logging
        dims=len(ext_obj)
        if binary:
            x = m.addVars(range(dims),
                             vtype=GRB.BINARY,
                             obj=ext_obj,
                             name="x")
        else:
            x = m.addVars(range(dims),
                             vtype=GRB.CONTINUOUS,
                             obj=ext_obj,
                             name="x")
        m.update()
        nz=sparse.find(self.ineqs.transpose())
        if self.rhs.shape[0]==1:
            rhs=self.rhs.transpose().toarray()
        else:
            rhs=self.rhs.toarray()
        num_ineqs=max(nz[1])+1
        j=0
        for i in range(num_ineqs):#add ineqs from the sparse matrix
            expr=0
            while nz[1][j]==i:
                expr += nz[2][j]*x[nz[0][j]]
                j+=1
                if j==len(nz[1]):
                    break
            m.addConstr(expr >= int(rhs[i][0]))
        m.update()
        m.ModelSense=1
        return m
    
    '''solve a (mixed integer) linear program with respect to the own projection
    Parameters:
        - self: relevant for the projection to the original space
        - m: the gurobipy model to be solved
    Output: 
        - m.objVal: the objective value of the model
        - m.Runtime: the time gurobipy needed to solve the model
        - res: the optimal point'''
    def solveModel(self,m):
        m.optimize()
        x=m.getVars()
        if m.Status==GRB.OPTIMAL:
            print("Opt. Value=%s"%m.ObjVal)
            res_ext=[]
            for i in range(len(x)):
                res_ext.append(x[i].X)
            res = [np.dot(res_ext,i) for i in self.proj.transpose().toarray()]#project the solution down to the original space
            print(res)
            return (m.ObjVal,round(m.Runtime,4),res)
        return
    
    '''Add a constraint to a polytope constructed by a hierarchy as if it were part of the model before the hierarchy ran. This saves time in iterated separation.
    Parameters:
        - self: the extended formulation to which the constraint will be added
        - ineq_new: the incidence vector of the new constraint
        - rhs_new: the value of the rhs of the new ineq'''
    def addConstraint(self, ineq_new, rhs_new):
        ineqs=self.ineqs.toarray()
        num_ps=int(sum(ineqs[-2]))
        rowsToAdd=[]
        for i in range(num_ps):
            row=np.concatenate((np.zeros(i*len(ineq_new)),ineq_new,np.zeros((num_ps-i-1)*len(ineq_new)),np.zeros(i),np.array([-rhs_new]),np.zeros(num_ps-i-1)))
            rowsToAdd.append(row[None,:])
        ineqs=np.concatenate((np.concatenate(rowsToAdd),ineqs))
        rhs=self.rhs.toarray()
        rhs=np.concatenate((np.zeros(num_ps)[:,None],rhs))
        self.ineqs=sparse.coo_matrix(ineqs)
        self.rhs=sparse.coo_matrix(rhs)
    
'''The following function calculates the convex hull of the union of two extended formulations. The result is one extended formulation. The underlying algorithm was discovered by BALAS.
Parameters:
    - extendedFormulation1: first extended Formulation
    - extendedFormulation2: second extended Formulation
Output: 
    - union: extendedFormulation representing the union of the input'''
def balas(extendedFormulation1, extendedFormulation2):
    '''The input for this function are two extended formulations E_1 and E_2. They are given in the following form: E_i={P_i*z|A_i*z>=b_i}, where the dimension of P_i*z needs to be equal for i=1 and i=2. The resulting extended formulation then is given by this formula: R={P_1*y_1+P_2*y_2|A_1*y_1>=lambda_1*b_1,A_2*y_2>=lambda_2*b_2,lambda_1+lambda_2=1,lambda>=0}'''
    result_ineqs1=np.concatenate((extendedFormulation1.ineqs,np.zeros((extendedFormulation1.ineqs.shape[0],extendedFormulation2.ineqs.shape[1])),np.negative(extendedFormulation1.rhs[:,None]),np.zeros((extendedFormulation1.ineqs.shape[0],1))),axis=1)
    result_ineqs2=np.concatenate((np.zeros((extendedFormulation2.ineqs.shape[0],extendedFormulation1.ineqs.shape[1])),extendedFormulation2.ineqs,np.zeros((extendedFormulation2.ineqs.shape[0],1)),np.negative(extendedFormulation2.rhs[:,None])),axis=1)
    result_ineqs3=np.concatenate((np.zeros((4,extendedFormulation1.ineqs.shape[1]+extendedFormulation2.ineqs.shape[1])),np.array([[ 1.0, 1.0],[-1.0,-1.0],[ 1.0, 0.0],[ 0.0, 1.0]])),axis=1)
    result_ineqs=np.concatenate((result_ineqs1,result_ineqs2,result_ineqs3),axis=0)
    result_rhs=np.append(np.zeros(result_ineqs.shape[0]-4),np.array([ 1.0,-1.0, 0.0, 0.0]))
    result_proj=np.concatenate((extendedFormulation1.proj,extendedFormulation2.proj,np.zeros((2,extendedFormulation1.proj.shape[1]))),axis=0)
    union=extendedFormulation(result_ineqs, result_rhs, result_proj)
    return union

'''This function achieves the same result as the above defined balas-function, but it works on arbitrary sets of polygons. Every extended formulation fed into this algorithms is required to have a projection into the same lower-dimensional space. The function itself is just a large technical concatenation of matrices. The underlying algorithm can be found in the dissertation of WELTGE on p.15f.
Parameters: 
    - extendedFormulation_list: list of extended formulations
Output: 
    - extendedFormulation(): union of the input'''
def balas_list(extendedFormulation_list):
    result_ineqs_list=[]
    for i,ef in enumerate(extendedFormulation_list):
        result_ineqsI=ef.ineqs
        result_ineqsI2=-ef.rhs.transpose()
        for j,ef2 in enumerate(extendedFormulation_list):
            if j<i:
                result_ineqsI=sparse.construct.hstack([sparse.coo_matrix((ef.ineqs.shape[0],ef2.ineqs.shape[1])),result_ineqsI])
                result_ineqsI2=sparse.construct.hstack([sparse.coo_matrix((ef.ineqs.shape[0],1)),result_ineqsI2])
            elif j>i:
                result_ineqsI=sparse.construct.hstack([result_ineqsI,sparse.coo_matrix((ef.ineqs.shape[0],ef2.ineqs.shape[1]))])
                result_ineqsI2=sparse.construct.hstack([result_ineqsI2,sparse.coo_matrix((ef.ineqs.shape[0],1))])
        result_ineqs_list.append(sparse.hstack([result_ineqsI,result_ineqsI2]))
    result_ineqsL1=sparse.coo_matrix((len(extendedFormulation_list)+2,sum([x.ineqs.shape[1] for x in extendedFormulation_list])))
    result_ineqsL2=sparse.construct.vstack([sparse.construct.identity(len(extendedFormulation_list)),sparse.coo_matrix(np.ones((1,len(extendedFormulation_list)))),-sparse.coo_matrix(np.ones((1,len(extendedFormulation_list))))])
    result_ineqs_list.append(sparse.construct.hstack([result_ineqsL1,result_ineqsL2]))
    result_ineqs=sparse.construct.vstack(result_ineqs_list)
    result_rhs=sparse.construct.vstack([sparse.coo_matrix((result_ineqs.shape[0]-2,1)),1,-1])
    result_proj=sparse.construct.vstack([x.proj for x in extendedFormulation_list]+[sparse.coo_matrix((len(extendedFormulation_list),extendedFormulation_list[0].proj.shape[1]))])
    return extendedFormulation(result_ineqs,result_rhs,result_proj)

'''The following function calculates the intersection of exactly two extended formulations. The result is one extended formulation. Therefore, both extended formulations are viewed in seperate dimensions and all inequalities must hold for the respective extended formulations. Additionally, the projections must be equal (i.e. projection of the first extendedFormulation - projection of the second extendedFormulation = 0). Thus, the resulting projection can wlog be set to the projection of the first extendedFormulation.
Parameters: 
    -extendedFormulation1: first extended formulation
    -extendedFormulation2: second extended formulation
Output:
    -extendedFormulation(): intersection of the extended formulations'''
def intersect(extendedFormulation1, extendedFormulation2):
    result_ineqs1=np.concatenate((extendedFormulation1.ineqs,np.zeros((extendedFormulation1.ineqs.shape[0],extendedFormulation2.ineqs.shape[1]))),axis=1)
    result_ineqs2=np.concatenate((np.zeros((extendedFormulation2.ineqs.shape[0],extendedFormulation1.ineqs.shape[1])),extendedFormulation2.ineqs),axis=1)
    result_ineqs3=np.concatenate((extendedFormulation1.proj.transpose(),np.negative(extendedFormulation2.proj.transpose())),axis=1)
    result_ineqs4=np.concatenate((np.negative(extendedFormulation1.proj.transpose()),extendedFormulation2.proj.transpose()),axis=1)
    result_ineqs=np.concatenate((result_ineqs1,result_ineqs2,result_ineqs3,result_ineqs4),axis=0)
    result_rhs=np.concatenate((extendedFormulation1.rhs,extendedFormulation2.rhs,np.zeros(2*extendedFormulation1.proj.shape[1])))
    result_proj=np.concatenate((extendedFormulation1.proj,np.zeros((extendedFormulation2.ineqs.shape[1],extendedFormulation1.proj.shape[1]))),axis=0)
    return extendedFormulation(result_ineqs,result_rhs,result_proj)

'''The following function calculates the intersection of exactly a list of extended formulations. The result is one extended formulation. The algorithm works analogous to the case for two extended formulations.
Parameters: 
    -extendedFormulation_list: list of extended formulations
Output:
    -extendedFormulation(): intersection of the extended formulations'''
def intersect_list(extendedFormulation_list):
    result_ineqs_lists1=[[sparse.coo_matrix((ef.ineqs.shape[0],ef2.ineqs.shape[1])) if i2!=i else ef.ineqs for i2,ef2 in enumerate(extendedFormulation_list)] for i,ef in enumerate(extendedFormulation_list)]
    result_ineqs_list1=[sparse.construct.hstack(x) for x in result_ineqs_lists1]
    result_ineqs1=sparse.construct.vstack(result_ineqs_list1)
    result_ineqs_lists2=[[ef2.proj.transpose() if i2==0 else -ef2.proj.transpose() if i+1==i2 else sparse.coo_matrix(ef2.proj.transpose().shape) for i2,ef2 in enumerate(extendedFormulation_list)] for i,ef in enumerate(extendedFormulation_list[1:])]
    result_ineqs_list2=[sparse.construct.hstack(x) for x in result_ineqs_lists2]
    result_ineqs2=sparse.construct.vstack(result_ineqs_list2)
    result_ineqs3=-result_ineqs2
    result_ineqs=sparse.construct.vstack((result_ineqs1,result_ineqs2,result_ineqs3))
    result_rhs=sparse.construct.vstack([x.rhs for x in extendedFormulation_list]+[sparse.coo_matrix((2*(len(extendedFormulation_list)-1)*extendedFormulation_list[0].proj.shape[1],1))])
    result_proj=sparse.construct.vstack([extendedFormulation_list[0].proj]+[sparse.coo_matrix(ef.proj.shape) for ef in extendedFormulation_list[1:]])
    return extendedFormulation(result_ineqs,result_rhs,result_proj)

'''The following function intersects an extended formulation with an inequality of the sort x_i>=1 or x_i<=0 from the original space.
Parameters: 
    -ef: the extended formulation
    -index: index of the variable which is set to 0 or 1
    - value: if value is equal to 1, the inequality x_i>=1 is added, else the inequality x_i<=0 is added.
Output:
    -extendedFormulation(): the resulting extended formulation'''
def intersect_hyperplane(ef, index, value):
    hyperplane=np.zeros(ef.proj.shape[0])
    if value==1:
        hyperplane[index]=1
        intersection_rhs=sparse.construct.hstack([ef.rhs,sparse.coo_matrix(1)])
    else:
        hyperplane[index]=-1
        intersection_rhs=sparse.construct.hstack([ef.rhs,sparse.coo_matrix(0)])
    hyperplane=sparse.coo_matrix(hyperplane)
    hyperplane=ef.proj.dot(hyperplane.transpose())#project hyperplane to higher-dimensional space
    intersection=sparse.construct.vstack([ef.ineqs.copy(),hyperplane.transpose()])#add hyperplane to inequalities of the approximation
    return extendedFormulation(intersection, intersection_rhs, ef.proj)