from __future__ import division
import itertools
import numpy as np

# from memory_profiler import profile

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


class MMCMultiClassQueueingSystem:
    def __init__(self, arrivalRates, serviceRates, nServers):
        assert len(arrivalRates)==len(serviceRates)
        self.lmbda=np.array(arrivalRates)
        self.mu=np.array(serviceRates)
        self.mClasses = len(arrivalRates)
        self.nServers = nServers
        assert np.sum(self.lmbda/self.mu) < nServers

        self.lambdaTot = sum(self.lmbda)
        self.alpha = self.lmbda/self.lambdaTot

        self.i_map_full = []
        self.idx_map_full = []

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Not Implemented"

    def computeZMatrix(self, eps = 0.00000001):

        #create index search dictionary
        idx_map = dict([(tuple(vect), i) for i, vect in zip(itertools.count(), generateVectorsFixedSum(self.mClasses, self.nServers))])
        i_map   = dict([(idx_map[idx], list(idx)) for idx in idx_map ])
        q_max = len(idx_map)


                                                    #identity matrix will corrspond to terms with i-1 items in queue
        #A0 = sps.dok_matrix((q_max,q_max))  #corresponds to terms with i items in queue
        #A1 = sps.dok_matrix((q_max,q_max))  #corresponds to terms with i+1 items in queue
        A0 = np.zeros((q_max,q_max))  #corresponds to terms with i items in queue
        A1 = np.zeros((q_max,q_max))  #corresponds to terms with i+1 items in queue

        #form generator matrix
        for i, idx in i_map.items(): #range(q_max):
            #diagonal term
            A0[i,i] += 1 + np.sum(idx*self.mu)/self.lambdaTot

            #term corresponding to end of service for item j1, start of service for j2
            for j1 in xrange(self.mClasses):
                for j2 in xrange(self.mClasses):
                    idx[j1] += 1; idx[j2] -= 1
                    i1 = getIndexDict(idx,idx_map)
                    if i1>=0: A1[i,i1] += self.alpha[j2]*idx[j1]*self.mu[j1]/self.lambdaTot
                    idx[j1] -= 1; idx[j2] += 1

        #solve the eigenvalue equation iteratively
        I = np.eye(q_max)
        Z_prev = np.zeros((q_max, q_max)) #I
        delta=1
        A0_inv = np.linalg.inv(A0)
        while delta>eps:
            Z = np.dot(A0_inv, I + np.dot(A1, np.dot(Z_prev, Z_prev)))  #invA0*(I+A1*H*H)
            delta = np.sum(np.abs(Z-Z_prev))
            Z_prev=Z

        self.Z = Z[:]

        self.idx_map_full.append(idx_map)
        self.i_map_full.append(i_map)

    def computeQMatrices(self):

        self.Q = []
        self.Q.insert(0, self.Z)

        idx_map_nplus = self.idx_map_full[-1]
        i_map_nplus   = self.i_map_full[-1]
        q_max_nplus   = len(idx_map_nplus)

        idx_map_n = idx_map_nplus
        i_map_n   = i_map_nplus
        q_max_n   = q_max_nplus

        A1_n = np.zeros((q_max_n,q_max_nplus))  #corresponds to terms with i+1 items in queue
        #form generator matrix
        for i, idx in i_map_n.items(): #range(q_max):
            #term corresponding to end of service for item j1, start of service for j2
            for j1 in xrange(self.mClasses):
                for j2 in xrange(self.mClasses):
                    idx[j1] += 1; idx[j2] -= 1
                    i1 = getIndexDict(idx,idx_map_n)
                    if i1>=0: A1_n[i,i1] += self.alpha[j2]*idx[j1]*self.mu[j1]/self.lambdaTot
                    idx[j1] -= 1; idx[j2] += 1

        for n in range(self.nServers,0,-1):
            idx_map_nminus = dict([ (tuple(vect), i) for i, vect in zip(itertools.count(), generateVectorsFixedSum(self.mClasses, n-1)) ])
            i_map_nminus   = dict([(idx_map_nminus[idx], list(idx)) for idx in idx_map_nminus ])
            q_max_nminus   = len(idx_map_nminus)

            self.idx_map_full.insert(0,idx_map_nminus)
            self.i_map_full.insert(0,i_map_nminus)


            L_n = np.zeros((q_max_n,q_max_nminus))  #corresponds to terms with i items in queue
            A0_n = np.zeros((q_max_n,q_max_n))  #corresponds to terms with i items in queue
            for i, idx in i_map_n.items():

                #diagonal term
                A0_n[i,i] += 1 + np.sum(idx*self.mu)/self.lambdaTot

                #term corresponding to arrival of item item j1
                for j2 in xrange(self.mClasses):
                    idx[j2] -= 1
                    i2 = getIndexDict(idx,idx_map_nminus)
                    if i2>=0: L_n[i,i2] += self.alpha[j2]
                    idx[j2] += 1

            # Q_n = (A_0 - A_1*Q_{n+1})^{-1}*L_n
            self.Q.insert(0, np.dot(np.linalg.inv(A0_n-np.dot(A1_n, self.Q[0])), L_n))

            idx_map_nplus = idx_map_n
            i_map_nplus   = i_map_n
            q_max_nplus   = q_max_n

            idx_map_n = idx_map_nminus
            i_map_n   = i_map_nminus
            q_max_n   = q_max_nminus

            A1_n = np.zeros((q_max_n,q_max_nplus))  #corresponds to terms with i+1 items in queue
            for i, idx in i_map_n.items():
                #term corresponding to end of service for item j1
                for j1 in xrange(self.mClasses):
                    idx[j1] += 1
                    i1 = getIndexDict(idx,idx_map_nplus)
                    if i1>=0: A1_n[i,i1] += idx[j1]*self.mu[j1]/self.lambdaTot
                    idx[j1] -= 1

    def computeP0andJointProbabilitites(self):
        self.P = []

        self.P.append(np.array([1.0]))

        sm = 0.0

        for n in xrange(self.nServers):
            sm += sum(self.P[-1])
            self.P.append(np.dot(self.Q[n],self.P[-1]))

        sm += sum(np.dot(np.linalg.inv(np.eye(len(self.P[-1])) - self.Z),self.P[-1]))

        for p in self.P:
            p /= sm



        pass

    def computeEQ(self):
        inv1minZ = np.linalg.inv(np.eye(len(self.P[-1])) - self.Z)
        self.EQTotal = sum(np.dot(np.dot(np.dot(inv1minZ,inv1minZ), self.Z),self.P[-1]))
        # self.EQTotal = sum(np.dot(np.dot(inv1minZ,inv1minZ),self.P[-1]))
        self.EQQmin1Total = 2*sum(np.dot(np.dot(np.dot(np.dot(np.dot(inv1minZ,inv1minZ),inv1minZ), self.Z), self.Z),self.P[-1]))
        self.EQ2Total = self.EQQmin1Total + self.EQTotal

        self.EQ = self.alpha*self.EQTotal
        self.EQQmin1 = self.alpha*self.alpha*self.EQQmin1Total
        self.EQ2 = self.EQQmin1 + self.EQ


        pass

    def computeVarQ(self):
        self.VarQTotal = self.EQ2Total - self.EQTotal**2
        self.VarQ = self.EQ2 - self.EQ**2

        pass

    def computeEN(self):
        # raise "not implemented"
        self.ENTotal = self.EQTotal + sum(self.lmbda/self.mu)
        self.EN = self.EQ + self.lmbda/self.mu


        self.ES2 = 0
        self.ESq = 0

        self.EN2 = self.EQ2 + self.lmbda/self.mu

        pass

    def computeVarN(self):
        raise "not implemented"
        pass

    def computeMarginalDistributions(self, qmax=500):
        self.marginalN = np.zeros((self.mClasses, qmax))

        for m in xrange(self.mClasses):
            for map, p in zip(self.i_map_full[:-1], self.P[:-1]):
                for i, idx in map.items():
                    self.marginalN[m,idx[m]] += p[i]

            inv1minAlphaZ = np.linalg.inv(np.eye(len(self.P[-1])) - (1-self.alpha[m])*self.Z)
            frac = np.dot(self.alpha[m]*self.Z, inv1minAlphaZ)
            # tmp = np.dot(self.Z, self.P[-1])
            # tmp = np.dot(inv1minAlphaZ, tmp)
            tmp = np.dot(inv1minAlphaZ, self.P[-1])

            for q in xrange(0,qmax):
                for i, idx in self.i_map_full[-1].items():
                    if idx[m]+q < qmax: self.marginalN[m,idx[m]+q] += tmp[i]
                tmp = np.dot(frac, tmp)


if __name__ == '__main__':
    np.set_printoptions(linewidth=50000)

    qmax = 500

    # failureRates = np.array([3, 4, 5])
    # serviceRates = np.array([5, 5, 5])
    # skillServerAssignment = np.array([[1,1,0],[1,0,1],[0,1,1]])

    failureRates = np.array([3, 4*1.1])
    serviceRates = np.array([10, 10*1.1])
    skillServerAssignment = np.array([[1,1]])

    nServers = skillServerAssignment.shape[0]
    mClasses = skillServerAssignment.shape[1]

    # failureRates = np.array([0.9])
    # serviceRates = np.array([1.0/3])
    # skillServerAssignment = np.array([[1],[1],[1]])
    #

    # qs = MMCMultiClassQueueingSystem([3.6], [1.0], 4)
    # qs = MMCMultiClassQueueingSystem([0.9, 0.9], [2.0, 2.0], 1)
    # qs = MMCMultiClassQueueingSystem([1.3, 0.5], [2.0/3, 1.0/3], 4)
    # qs = MMCMultiClassQueueingSystem([0.9], [1.0], 1)

    qs = MMCMultiClassQueueingSystem(failureRates, serviceRates, skillServerAssignment.shape[0])

    qs.computeZMatrix(eps=0.000000000000000001)

    qs.computeQMatrices()

    qs.computeP0andJointProbabilitites()

    qs.computeMarginalDistributions(qmax=qmax)

    qs.computeEQ()
    qs.computeEN()

    print [qs.marginalN[i,10]/qs.marginalN[i,11]*failureRates[i]/serviceRates[i] for i in range(mClasses)]
    print [nServers-sum((failureRates/serviceRates)[list(set(range(mClasses)) - {i})]) for  i in range(mClasses)]
    # print (1-(failureRates/serviceRates)[1]/skillServerAssignment.shape[0])*skillServerAssignment.shape[0], \
    #         (1-(failureRates/serviceRates)[0]/skillServerAssignment.shape[0])*skillServerAssignment.shape[0]

    # print [qs.marginalN[i,10]/qs.marginalN[i,11]*failureRates[i]/serviceRates[i] for i in range(2)]

    alpha = np.ones(skillServerAssignment.shape)/skillServerAssignment.shape[0]

    delta = 1-alpha*failureRates/serviceRates

    import sys
    sys.path.insert(0, '/Users/andrei/Documents/Research/Research Papers/Skill Assignment/PythonCodes/')
    from simulation_codes import simulation_batch_run
    from simulation_codes import generateSkillsMultiServer, checkSkillAssignment


    # from simulation import simulation_batch_run
    # from QueueingModels import ComputeMarginalDistributions
    #
    # marginalDistributionSingleServer = ComputeMarginalDistributions(failureRates, serviceRates)




    marginalSimulation, _, utilizationRates, skillServerAssignmentStatisitcs = \
        simulation_batch_run(failureRates,
                         serviceRates,
                         skillServerAssignment,
                         expectedTotalNumberRequests = 100000,
                         numberWarmingUpRequests = 0,
                         replications=35,
                         maxQueue=500)


    print [marginalSimulation[i,10]/marginalSimulation[i,11]*failureRates[i]/serviceRates[i] for i in range(mClasses)]


    # qs.



    # requestArrivalRate = np.sum(failureRates)
    # skillDistribution = np.cumsum(failureRates/requestArrivalRate)
    # # skillServerCosts = np.ones(skillServerAssignment.shape)
    # skillServerRates = np.transpose(np.zeros(skillServerAssignment.shape))
    # for i in xrange(serviceRates.shape[0]): skillServerRates[i,:] = skillServerAssignment[:,i] * serviceRates[i]
    # skillServerRates = np.transpose(skillServerRates)
    # skillServerCosts = skillServerRates
    #
    # numberInSystemDistribution, _, _, _, =\
    #     simulation(requestArrivalRate, skillDistribution, skillServerRates, skillServerCosts,
    #            histogramBucketsInput = [],
    #            expectedTotalNumberRequests = 1000000,
    #            numberWarmingUpRequests = 0,
    #            seed = 12345)
    #
    # marginalSimulation = np.zeros((len(failureRates), qmax))
    # for sk in numberInSystemDistribution:
    #     for n in numberInSystemDistribution[sk]:
    #         if(n < qmax):
    #             marginalSimulation[sk,n] += numberInSystemDistribution[sk][n]


    # qs.computeEN()

    # print np.sum(np.abs(qs.marginalN - marginalSimulation))
    pass

#print optimize_stock_server(12, 6.0, 10, 100, 1000)

    #pass