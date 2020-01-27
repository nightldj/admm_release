# -*- coding: utf-8 -*-
"""
ACADMM code
@author: Zheng Xu, xuzhustc@gmail.com
@reference: Xu et al. 2017, Adaptive Consensus ADMM for Distributed Optimization
"""

from mpi4py import MPI
import numpy as np
import time

class ACADMM(object):  #base class for ACADMM
    def __init__(self, adp_flag, v0, l0):
        #MPI related
        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_r = self.mpi_comm.Get_rank()

        #admm parameters
        self.maxiter = 100 #maximum iterations
        self.tol = 1e-3 #tolerance, stop criterion
        self.tau = 1 #initial penalty parameter, local
        self.adp_flag = adp_flag #adaptive flag, local,
        self.adp_freq = 2 #update frequency
        self.corr_thre = 0.2 #safeguarding: correlation threshold
        self.const_cg = 1e5
        self.minval = 1e-20 #minimum value to avoid NaN
        self.verbose = True #print intermediate results
        self.stop = False #stop flag

        #initialization
        self.u = None #local u
        v0 = self.mpi_comm.bcast(v0, root=0) #broadcast global v0
        self.v = v0.copy() #gloabal v
        self.l = l0.copy() #local l (dual)
        self.hl = self.l.copy() #local interemediate l (dual)
        self.v1 = self.v.copy()
        self.pres = 0 #local primal residual
        self.dres = 0 #local dual residual
        self.obj = 0
        if self.mpi_r == 0:
            self.g_pres = 0 #global primal residual
            self.g_dres = 0 #global dual residual
        self.all_u = None
        self.all_l = None
        self.all_hl = None


    def update_u(self): #update local u, create memory, do not use inplace u
        pass

    def update_v(self): #update global v, create memory, do not use inplace v
        pass


    def compute_obj(self):
        pass

    def shrink(self, v, t):
        return np.sign(v)*np.maximum(np.abs(v)-t, 0)

    def update_residual(self):
        #local residual
        pres = np.linalg.norm(self.v-self.u, 'fro')**2
        self.pres=pres
        dres = (self.tau*np.linalg.norm(self.v1-self.v, 'fro'))**2
        self.dres=dres
        #local relative value
        res_u = np.linalg.norm(self.u, 'fro')**2
        res_l = np.linalg.norm(self.l, 'fro')**2

        #global residual
        pres = self.mpi_comm.gather(pres, root=0)
        dres = self.mpi_comm.gather(dres, root=0)
        res_u = self.mpi_comm.gather(res_u, root=0)
        res_l = self.mpi_comm.gather(res_l, root=0)
        if self.mpi_r == 0: #central server
            self.g_pres = np.mean(pres) #global is mean of local
            res_u = np.mean(res_u)
            res_v = np.linalg.norm(self.v, 'fro')**2
            self.g_dres = np.mean(dres)
            res_l = np.mean(res_l)
            prel = res_u if res_u > res_v else res_v
            self.rel_tol = max([self.g_pres/max([prel, self.minval]), self.g_dres/max([res_l, self.minval])]) #relative residual
            if self.verbose:
                print 'iter: %d, relres: %f, pres: %f/%f, dres:%f/%f'%(self.iter, self.rel_tol, self.g_pres, prel, self.g_dres, res_l)
            if self.rel_tol < self.tol: #stop criterion
                self.stop = True
        self.stop = self.mpi_comm.bcast(self.stop, root=0)



    def fuse_spectral(self, sd, mg):
        return mg if mg > 0.5*sd else sd-0.5*mg


    def local_spectral(self):
        if 0 == self.iter: #first iteration, cannot estimate, record
            self.hl += self.tau*(self.v1-self.u) #local update l
            self.u0 = self.u
            self.l0 = self.l.copy()
            self.hl0 = self.hl.copy()
            self.v0 = self.v
        elif self.iter % self.adp_freq == 1: #estimate adaptive
            self.hl += self.tau*(self.v1-self.u) #local update l
            #spectral alpha
            dhl = self.hl - self.hl0
            du = self.u - self.u0
            dhlsq = np.mean(dhl*dhl)
            duhl = np.mean(du*dhl)
            dusq = np.mean(du*du)
            #spectral beta
            dl = self.l - self.l0
            dv = self.v0 - self.v
            dlsq = np.mean(dl*dl)
            dvsq = np.mean(dv*dv)
            dvl = np.mean(dv*dl)

            #safeguarding
            if duhl > self.corr_thre*np.sqrt(dusq)*np.sqrt(dhlsq) + self.minval:
                ha_flag = True
            else:
                ha_flag = False
            if dvl > self.corr_thre*np.sqrt(dvsq)*np.sqrt(dlsq) + self.minval:
                hb_flag = True
            else:
                hb_flag = False

            if ha_flag:
                ha = self.fuse_spectral(dhlsq/duhl, duhl/dusq)
            else:
                ha = 0
            if hb_flag:
                hb = self.fuse_spectral(dlsq/dvl, dvl/dvsq)
            else:
                hb = 0

            #update tau
            tau0 = self.tau
            if ha_flag and hb_flag:
                self.tau = np.sqrt(ha*hb)
            elif ha_flag:
                self.tau = ha
            elif hb_flag:
                self.tau = hb

            #if self.verbose and self.mpi_r == 0:
                #print 'rank:%d, iter:%d, duhl:%f, dusq:%f, dhlsq:%f, dvl:%f, dvsq:%f, dlsq:%f'%(self.mpi_r, self.iter, duhl, dusq, dhlsq, dvl, dvsq, dlsq)
                #print 'rank:%d, iter:%d, ha:%f(%d), hb: %f(%d), tau:%f'%(self.mpi_r, self.iter, ha, ha_flag, hb, hb_flag, self.tau)

            #bound penalty for convergence
            bd = 1.0 + self.const_cg/float(self.iter+1)**2
            self.tau = max(min(self.tau, tau0*bd), tau0/bd)


            #record
            self.u0 = self.u
            self.l0 = self.l.copy()
            self.hl0 = self.hl.copy()
            self.v0 = self.v


    def optimize(self):
        stime = time.time()
        if self.mpi_r ==0:
            print 'ACADMM, adp_flag:%d, init tau:%f, maxiter:%d, reltol:%f, corr_thre:%f, const_cg:%f'%(self.adp_flag, self.tau, self.maxiter, self.tol, self.corr_thre, self.const_cg)

        for i in xrange(self.maxiter):
            self.iter = i

            #ADMM steps
            self.update_u() #local update u, customize for different problem
            self.update_v() #global update v, customize for different problem
            self.l += self.tau*(self.v-self.u) #local update l

            #residual and stop criterion
            self.update_residual()
            self.compute_obj() #objective
            if self.verbose and self.mpi_r == 0:
                print 'iter: %d, obj:%f'%(self.iter, self.obj)
            if self.stop:
                break

            #adaptive penalty
            if self.adp_flag:
                self.local_spectral()

            #show adaptive penalty
            self.tau = max(self.tau, self.minval)
            if self.verbose:
                #all_tau = self.tau
                #all_tau = self.mpi_comm.gather(all_tau, root=0) #defined in update_v()
                if self.mpi_r == 0:
                    print 'iter:%d, tau mean:%f, std:%f'%(i, np.mean(self.all_tau), np.std(self.all_tau))
                    print 'iter:%d, time:%f, relres:%f'%(i, time.time()-stime, self.rel_tol)

            #previous iterates
            self.v1 = self.v

        if self.mpi_r == 0:
            print 'complete optimize, adp_flag:%d, iter:%d, time:%f, relres %f'%(self.adp_flag, i, time.time()-stime, self.rel_tol)



    def load_data(self, D, c, augFlag = False): #rewrite to customize
        outD, outc = [], []
        if self.mpi_r == 0: #central server to distributed data
            #data
            n,d = D.shape
            assert(n==c.shape[0]) #number of sample match label
            nb = n//self.mpi_n #block size per node
            for r in xrange(self.mpi_n):
                if r == self.mpi_n - 1: #last block
                    outD.append(D[r*nb:])
                    outc.append(c[r*nb:])
                else:   #r block
                    outD.append(D[r*nb:(r+1)*nb])
                    outc.append(c[r*nb:(r+1)*nb])
            print 'App L1Logreg: samples: %d, ftrs:%d, blocksize:%d, nodes:%d'%(n,d,nb,self.mpi_n)
        outD = self.mpi_comm.scatter(outD, root=0)
        outc = self.mpi_comm.scatter(outc, root=0)
        self.D = outD
        self.c = outc
        n,d = self.D.shape
	if n < d:
            self.Dsq = self.D.dot(self.D.T)
	else:
	    self.Dsq = self.D.T.dot(self.D)
        self.Dtc = self.D.T.dot(self.c)




class ACADMM_ENReg(ACADMM):
    def __init__(self, r1, r2, adp_flag, v0, l0):
        #init ACADMM
        super(ACADMM_ENReg, self).__init__(adp_flag, v0, l0)
        self.mpi_n = self.mpi_comm.Get_size() #number of nodes
        #app parameter
        self.r1 = float(r1)
        self.r2 = float(r2)
        if self.mpi_r == 0:
            print 'Elastic net regression, r1: %f, r2: %f'%(r1, r2)



    def update_u(self): #update local u, create memory, do not use inplace u
        n,d = self.D.shape
	#if self.mpi_r == 0:
	    #print "update u", n,d
        if n < d:
	    tmp = self.v+(self.l+self.Dtc)/self.tau
	    self.u = tmp - self.D.T.dot(np.linalg.inv(self.tau*np.eye(n)+self.Dsq).dot(self.D.dot(tmp)))
        else:
            self.u = np.linalg.inv(self.Dsq+self.tau*np.eye(d)).dot(self.tau*self.v+self.l+self.Dtc)


    def update_v(self): #update global v, create memory, do not use inplace v
	#if self.mpi_r == 0:
	    #print "update v start"
        all_tau = self.tau
        all_tau = self.mpi_comm.gather(all_tau, root=0)
        all_tau = np.array(all_tau)
        self.all_tau = all_tau
        all_u = self.u
        all_u = self.mpi_comm.gather(all_u, root=0)
        all_u = np.array(all_u)
        self.all_u = all_u
        all_l = self.l
        all_l = self.mpi_comm.gather(all_l, root=0)
        all_l = np.array(all_l)
        self.all_l = all_l
        if self.mpi_r == 0:
            tmp = np.array([self.all_u[i]*self.all_tau[i] for i in xrange(self.mpi_n)])
            self.v = self.shrink(np.mean(tmp-self.all_l, axis=0), self.r1/self.mpi_n)/(np.mean(self.all_tau)+self.r2/self.mpi_n)
        self.v = self.mpi_comm.bcast(self.v, root=0)
	#if self.mpi_r == 0:
	    #print "update v end"


    def compute_obj(self):
        obj = 0.5*np.linalg.norm(self.D.dot(self.u)-self.c, 2)**2
        obj = self.mpi_comm.gather(obj, root=0)
        if self.mpi_r == 0:
            obj = np.sum(obj)
            obj += self.r1*np.linalg.norm(self.v, 1) + 0.5*self.r2*np.linalg.norm(self.v, 2)**2
            self.obj = obj
        else:
            self.obj = None


