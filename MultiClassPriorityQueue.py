'''
Created on 26 Apr 2015

@author: andrei
'''
import itertools
import numpy as np
import scipy.sparse as sps
import scipy.misc as msc


def getIndexDict(idx, idx_map):
    try:
        return idx_map[tuple(idx)]
    except KeyError:
        return -1
    
def generateVectorsFixedSum(m,n):
    if m==1:
        yield [n]
    else:
        for i in range(n+1):
            for vect in generateVectorsFixedSum(m-1,n-i):
                yield [i]+vect
                
                
class QueueinSystem(object):
    '''
    classdocs
    '''
    def __init__(self, nClasses, nServers, lmbda, mu, priorities):
        '''
        Constructor
        '''
        self.nClasses = nClasses
        self.nServers = nServers
        self.lmbda = lmbda
        self.mu = mu
        self.priorities = priorities
        
    def computeZMatrix(self):
        
        #create index search dictionary
        idx_map = dict([ (tuple(vect), i) for i, vect in zip(itertools.count(), generateVectorsFixedSum(self.nClasses, self.nServers)) ])    
        i_map   = dict([(idx_map[idx], list(idx)) for idx in idx_map ])
        #q_max = len(idx_map)

        
        #A0 = sps.dok_matrix((q_max,q_max))  #corresponds to terms with i items in queue
        #A1 = sps.dok_matrix((q_max,q_max))  #corresponds to terms with i+1 items in queue
        A0 = np.array((len(idx_map),len(idx_map)))  #corresponds to terms with i items in queue
        A1 = np.array((len(idx_map),len(idx_map)))  #corresponds to terms with i+1 items in queue
            
        lambdaTot = sum(self.lmbda)
        #alpha = self.lmbda/lambdaTot
        #rho = self.lmbda/self.mu/self.nServers

        #form generator matrix
        for i, idx in i_map: #range(q_max):
            #idx = i_map[i]
            
            #diagonal term
            A0[i,i] += 1 + np.sum(np.multiply(idx, self.mu))/lambdaTot
            
            #term corresponding to end of service for item j1, start of service for j2 
            for j1 in range(self.nClasses):
                for j2 in range(self.nClasses):
                    idx[j1] += 1; idx[j2] -= 1
                    i1 = getIndexDict(idx,idx_map)
                    if i1>=0: A1[i,i1] += self.r[j2]*idx[j1]*self.mu[j1]/lambdaTot
                    idx[j1] -= 1; idx[j2] += 1

        #solve the eigenvalue equation iterativelA
        I = np.diag(np.ones((len(idx_map))))
        Z_prev = I
        delta=1
        while delta>0.000001:
            Z = np.dot(np.linalg.inv(A0), I + np.dot(A1, np.dot(Z_prev, Z_prev)))  #invA0*(I+A1*H*H)
            delta = np.sum(np.abs(Z-Z_prev)) 
            Z_prev=Z
    
            
    
    def computeQMatrices(self):
        
        idx_map_k = dict([ (tuple(vect), i) for i, vect in zip(itertools.count(), generateVectorsFixedSum(self.nClasses, self.nServers)) ])    
        i_map_k   = dict([(idx_map_k[idx], list(idx)) for idx in idx_map_k ])
        
        F={}
        for k in range(self.nServers, 0, -1):
            idx_map_kmin1 = dict([ (tuple(vect), i) for i, vect in zip(itertools.count(), generateVectorsFixedSum(self.nClasses, k-1)) ])    
            i_map_kmin1   = dict([(idx_map_kmin1[idx], list(idx)) for idx in idx_map_kmin1 ])
            F[k]=np.array((len(idx_map_k),len(idx_map_kmin1)))
            for i, idx in idx_map_k: #range(q_max):
                pass
                
        
            
        a=123
    
        
            