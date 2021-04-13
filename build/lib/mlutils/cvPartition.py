# -*- coding: utf-8 -*-
"""
# t: type: 1: random K-fold (K=n is LOO); 2: stratified K-fold; 
  # c: if type==2, c must be provided as a binary vector for the Positive/Negative classes
  # n: total n for parition; if t==2, this n is ignored
  # K: K-fold
  # r: repeat r times

for tr, te in cvPartition.foldPartition(c,len(c),10,2,2):
    print(np.sum(c[tr]))
    print(np.sum(c[te]))   
    
Created on Sat Jan 30 12:23:38 2021

@author: SYang
"""
import random
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr


def foldPartition(c, n, K=10, r=2, t=2):
    random.seed(3010)
    
    if t == 1:
        lenfold = n//K
        if K == n:
            lenfold = 1
    else: # stratified K-fold
        pidx = np.where(c==1)[0] # get element from a tuple
        nidx = np.where(c==0)[0]
        nump = len(pidx)
        numn = len(nidx)
        if nump >= K:
            plenfold = nump // K
        else:
            plenfold = np.int(np.ceil(nump / K)) # # of P per fold
        nlenfold = numn // K # # of N per fold
        #if maj and (plenfold+nlenfold > n):
        #    lenfold = n - plenfold - nlenfold
        #else:
        lenfold = plenfold + nlenfold
    
    #folds = np.zeros( (K*r, lenfold) )
    cvtools = importr('cvTools')
    rNumn = ro.vectors.IntVector([numn])
    rNump = ro.vectors.IntVector([nump])
    rK = ro.vectors.IntVector([K])
    rR = ro.vectors.IntVector([r])
    
    if t == 1:
        if n == K:
            folds = c
            return folds
        else:
            ftmp = cvtools.cvFolds(n, rK, rR)
            subsets = np.array(ftmp.rx2['subsets']) - 1
            rWhich = np.array(ftmp.rx2['which']) - 1
            for idx in range(r):
                ptmp = subsets[0:(K*lenfold), idx]
                for idy in range(K):
                    yield ptmp.iloc(rWhich[0:K*lenfold]==(idy+1))
                    
    else: # t == 2
        if nump <= K: # #positive case is less than K. expand the positive cases to sufficient length
            nps = int(np.ceil(nump / K))
            nPartition = cvtools.cvFolds(rNumn, rK, rR)
            subsets = np.array(nPartition.rx2['subsets']) - 1
            rWhich = np.array(nPartition.rx2['which']) - 1

            for idx in range(r):
                vecpidx = np.tile( np.random.permutation(pidx), int(np.ceil(nump/K)+1) )[0:K*nps]
                ntmp = subsets[0: (K*nlenfold), idx]
                for idy in range(K):
                    ptmp = vecpidx[ int(idy*nps) : int((idy+1)*nps) ]
                    ntemp = np.where(rWhich == idy)[0]
                    #if maj and ((plenfold+nlenfold)>n):
                    tmpTest = np.append( ptmp, nidx[ntmp[ntemp[0:nlenfold]]].astype(int) )
                    yield np.setdiff1d( range(n), tmpTest), tmpTest
                    #else:
                    #    yield np.append( ptmp, nidx[ntmp[ntemp[0:nlenfold]]].astype(int) )

        else:
            pPartition = cvtools.cvFolds(rNump, rK, rR)
            nPartition = cvtools.cvFolds(rNumn, rK, rR)
            pSubsets = np.array(pPartition.rx2['subsets']) - 1
            nSubsets = np.array(nPartition.rx2['subsets']) - 1
            pWhich = np.array(pPartition.rx2['which']) - 1
            nWhich = np.array(nPartition.rx2['which']) - 1
            print(r)
            for idx in range(r):
                ptmp = pSubsets[0:(K*plenfold+1), idx]
                ntmp = nSubsets[0:(K*nlenfold+1), idx] 
                for idy in range(K):
                    ptemp = np.where(pWhich == idy)[0]
                    ntemp = np.where(nWhich == idy)[0]
                    #if maj and ((plenfold+nlenfold)>n):
                    tmpTest = np.append( pidx[ptmp[ptemp[0:plenfold]]].astype(int), nidx[ntmp[ntemp[0:nlenfold]]].astype(int) )
                    yield np.setdiff1d( range(n), tmpTest), tmpTest
                    #else:
                    #    yield np.append( pidx[ptmp[ptemp[0:plenfold]]].astype(int), nidx[ntmp[ntemp[0:nlenfold]]].astype(int) )

                        
    #return folds