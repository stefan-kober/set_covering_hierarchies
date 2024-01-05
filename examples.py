'''This file contains several example set coverings and offers the possibility to generate random problems.'''

import numpy as np
from scipy import sparse

k3 = sparse.coo_matrix(np.array([[ 0.0, 1.0, 1.0], 
                                  [ 1.0, 0.0, 1.0], 
                                  [ 1.0, 1.0, 0.0]]))
k4 = sparse.coo_matrix(np.array([[1.0,1.0,0.0,0.0],
                                  [1.0,0.0,1.0,0.0],
                                  [1.0,0.0,0.0,1.0],
                                  [0.0,1.0,1.0,0.0],
                                  [0.0,1.0,0.0,1.0],
                                  [0.0,0.0,1.0,1.0]]))
c5 = sparse.coo_matrix(np.array([[1.0,1.0,0.0,0.0,0.0],
                                  [0.0,1.0,1.0,0.0,0.0],
                                  [0.0,0.0,1.0,1.0,0.0],
                                  [0.0,0.0,0.0,1.0,1.0],
                                  [1.0,0.0,0.0,0.0,1.0]]))
MaD5 = sparse.coo_matrix(np.array([[0,0,1,0,1],
                                     [0,1,0,1,1],
                                     [0,1,1,1,0],
                                     [1,0,0,1,0],
                                     [1,1,0,0,0]]))
axe = sparse.coo_matrix(np.array([[1,1,0,0,0],
                                    [0,1,1,0,0],
                                    [0,0,1,1,0],
                                    [0,0,1,0,1],
                                    [0,0,0,1,1]]))

'''This function generates a random graph with non-redundant edges.
Parameters:
    - nodes: number of nodes, n
    - edges: number of edges m
Output: 
    - matrix(): 0-1-matrix with n columns and m rows, node-edge-incidence-matrix of distinct edges'''
def generateRandomGraph(nodes, edges):
    if edges>nodes*(nodes-1)/2:
        raise ValueError("a graph with %s nodes can not have %s distinct edges."%(nodes,edges))
    edgeVecs=[]
    i=0
    while i<edges:
        added=False
        edgeVec=np.zeros(nodes)
        a = np.arange(nodes)
        np.random.shuffle(a)
        edgeVec[a[0]]=1
        edgeVec[a[1]]=1
        if not any(np.array_equal(edgeVec, e) for e in edgeVecs):
            edgeVecs.append(edgeVec)
            added=True
        if added:
            i+=1        
    g=np.concatenate([e[None,:] for e in edgeVecs], axis=0)
    return sparse.coo_matrix([tuple(row) for row in g])

'''This function generates several random graphs with non-redundant edges.
Parameters:
    - nodes: number of nodes, n
    - edges: number of edges, m
    - amount: number of graphs
Output: 
    - [generateRandomGraph]: list of node-edge-incidence-matrices'''
def generateRandomGraphs(nodes, edges, amount):
    return [generateRandomGraph(nodes, edges) for i in range(amount)]

'''This function generates a random set-covering-problem.
Parameters:
    - nodes: number of nodes, n
    - hedges: number of hyperedges, m
    - card: cardinality of a hyperedge
Output: 
    - matrix(): 0-1-matrix with n columns and m rows, node-hyperedge-incidence-matrix of the hypergraph. There may appear duplicates of hyperedges.'''
def generateRandomSCProblem(nodes, hedges, card):
    edgeVecs=[]
    for i in range(hedges):
        edgeVec=np.zeros(nodes)
        a = np.arange(nodes)
        np.random.shuffle(a)
        for i in range(card):
            edgeVec[a[i]]=1
        edgeVecs.append(edgeVec)
    g=np.concatenate([e[None,:] for e in edgeVecs], axis=0)
    return sparse.coo_matrix(g)