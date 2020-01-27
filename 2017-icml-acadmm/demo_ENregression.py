# -*- coding: utf-8 -*-
"""
demo, elastic net regularized regression
@author: Zheng Xu, xuzhustc@gmail.com
"""

import numpy as np
import acadmm
import datetime
import sys


def create_synthetic(N, F, seed=2017, num_active=1, noise=0, gmm=4):
    #N: number of samples, F:number of features, gmm:number of Gaussian for create GMM samples
    #random data from GMM
    np.random.seed(seed)
    phi = np.random.randn(N, F)
    if gmm > 1:
        nb = N//gmm
        for i in xrange(gmm):
            mu = np.random.randn(1, F)
            sg = np.abs(np.random.randn(1, F))
            phi[i*nb:(i+1)*nb, :] *= sg
            phi[i*nb:(i+1)*nb, :] += mu
    #groundtruth vector
    x = np.zeros((F, 1))
    x[0:num_active] = 1.0
    #print x

    #regression and noise
    b = phi.dot(x)
    b += np.random.normal(size=b.shape)*noise*np.mean(np.abs(b))
    return phi,b


if __name__ == "__main__":

    #dataset configure
    n = 100 #sample number
    d = 200 #sample dimension, feature number
    gmm = 4

    ad = max(1, int(d*0.1))  #active feature
    noise = 0.1   #noise level

    r1 = 10  #L1 regularizer
    r2 = r1 #L2 regularizer

    seed = 2017
    np.random.seed(seed)
    
    #optimizer configure
    v0 = np.random.randn(d, 1)
    l0 = np.ones((d, 1))
    m = acadmm.ACADMM_ENReg(r1=r1, r2=r2, adp_flag=True, v0=v0, l0=l0)


    if m.mpi_r == 0: #central server
        print 'start time:', datetime.datetime.now()
        D, c = create_synthetic(N=n, F=d, num_active=ad, seed=seed, noise=noise, gmm=gmm)
    else:
        D,c = None,None
    m.load_data(D,c)
    m.optimize()
    if m.mpi_r == 0:
        #print m.v
        print 'solution average: %f/%f'%(np.mean(m.v), float(ad)/d)
        print 'complete time:', datetime.datetime.now()
