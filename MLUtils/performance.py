#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:20:00 2020

@author: SYang
"""
# plot roc curve via r pROC package
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
#import matplotlib.font_manager as font_mgr


def plot_roc_curve_r(trY, pred, color='#ADDFFF',label=None,fig=None, ax = None, isLast = False):
    # fig, ax = plt.subplots()  # pass fig and ax to this function before use
    #loc1 = trY.isnull()
    #loc2 = pred.isnull()
    #locs =np.logical_not(np.logical_or(loc1,loc2))
    #trY = trY[locs]
    #pred = pred[locs]
    proc = importr('pROC')
    ground = ro.vectors.IntVector(trY)
    score1 = ro.vectors.FloatVector(pred)#[:,1])
    roc1 = proc.roc(ground, score1,direction='<', ci='True')  #roc1.names
    cilow = roc1.rx2('ci')[0] # low ci
    cihigh = roc1.rx2('ci')[2] # high ci
    auc = roc1.rx2('ci')[1] # auroc
    sensitivity = np.array(roc1.rx2('sensitivities'))
    specificity = np.array(roc1.rx2('specificities'))
    
    if ax is None:
        fig, ax = plt.subplots()
        isLast = True
    if label is None:
        ax.plot(1-specificity, sensitivity, color = color)
    else:
        label = label + ': AUC:' + "%.2f"%auc + ', 95%CI: '+"%.2f"%cilow + '-' + "%.2f"%cihigh
        ax.plot(1-specificity, sensitivity,color = color, label=label)
    
    if isLast:
        #font = font.mgr.FontProperties(family='', size)
        fig.legend(loc='lower right',bbox_to_anchor=(0.7, 0.15), frameon=False,prop={"family":'Consolas',"size":16})
        ident = [0.0, 1.0]
        ax.plot(ident,ident, color='#C0C0C0', linestyle='dashed')
        ax.set_xlabel('False positive rate', fontsize = 16)
        ax.set_ylabel('True positive rate', fontsize = 16)
    
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect('equal', adjustable='box')
        #fig.tight_layput()



def plot_prc_curve_r(trY, pred, color=None,label=None,fig=None, ax = None, isLast = False):
    # fig, ax = plt.subplots()  # pass fig and ax to this function before use
    loc1 = trY.isnull()
    loc2 = pred.isnull()
    locs =np.logical_not(np.logical_or(loc1,loc2))
    trY = trY[locs]
    pred = pred[locs]
    rocr = importr('ROCR')
    ground = ro.vectors.IntVector(trY)
    score1 = ro.vectors.FloatVector(pred)#[:,1])
    pre = rocr.prediction(score1, ground)
    pref = rocr.performance(pre, 'ppv') #tuple(pref.slotnames())
    precision = np.array(pref.slots['y.values'][0])
    pref = rocr.performance(pre, 'tpr')
    recall = np.array(pref.slots['y.values'][0])#[0]
    #remove nan
    locs = np.logical_not(np.isnan(precision))
    
    if ax is None:
        fig, ax = plt.subplots()
        isLast = True
    if color is None:
        ax.plot(recall[locs], precision[locs])
    else:
        ax.plot(recall[locs], precision[locs], color = color, label=label)
    
    if isLast:
        #font = font.mgr.FontProperties(family='', size)
        fig.legend(loc='lower right',bbox_to_anchor=(0.7, 0.15), frameon=False,prop={"family":'Consolas',"size":14})
        ident = [0.0, 1.0]
        ax.plot(ident,[1.0, 0.0], color='#C0C0C0', linestyle='dashed')
        ax.set_xlabel('Recall (TPR)', fontsize = 16)
        ax.set_ylabel('Precision (PPV)', fontsize = 16)
    
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect('equal', adjustable='box')
        #fig.tight_layput()

def plot_prc_curve_(trY, pred):
    rocr = importr('ROCR')
    ground = ro.vectors.IntVector(trY)
    score1 = ro.vectors.FloatVector(pred[:,1])
    pre = rocr.prediction(score1, ground)
    pref = rocr.performance(pre, 'ppv') #tuple(pref.slotnames())
    precision = np.array(pref.slots['y.values'][0])
    pref = rocr.performance(pre, 'tpr')
    recall = np.array(pref.slots['y.values'][0])#[0]
    #remove nan
    locs = np.logical_not(np.isnan(precision))
    plt.plot(recall[locs], precision[locs])
    plt.xlabel('Recall (TPR)', fontsize = 16)
    plt.ylabel('Precision (PPV)', fontsize = 16)
    plt.plot([0.0,1.0],[1.0,0.0], color='#C0C0C0', linestyle='dashed')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layput()
    
    
def plot_ppvnpv_(trY, pred):
    rocr = importr('ROCR')
    ground = ro.vectors.IntVector(trY)
    score1 = ro.vectors.FloatVector(pred)
    pre = rocr.prediction(score1, ground)
    pref = rocr.performance(pre, 'ppv') #tuple(pref.slotnames())
    cutoffs = np.array(pref.slots['x.values'][0])
    precision = np.array(pref.slots['y.values'][0])
    pref = rocr.performance(pre, 'npv')
    npv = np.array(pref.slots['y.values'][0])#[0]
    #remove nan
    locs = np.logical_not(np.isnan(precision))
    #plt.plot(recall[locs], precision[locs])
    fig, ax = plt.subplots()
    ax.plot(cutoffs[locs], precision[locs],label='PPV')
    cutoffs = np.array(pref.slots['x.values'][0])
    locs = np.logical_not(np.isnan(npv))
    ax.plot(cutoffs[locs], npv[locs],label='NPV')
    fig.legend(loc='lower right',bbox_to_anchor=(0.8, 0.15), frameon=False,prop={"family":'Consolas',"size":14})
    ax.set_xlabel('Score', fontsize = 16)
    ax.set_ylabel('PPV and NPV', fontsize = 16)
    #plt.plot([0.0,1.0],[1.0,0.0], color='#C0C0C0', linestyle='dashed')
    #plt.xlim(0, 1)
    #plt.ylim(0, 1)
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.tight_layout()
    
    
def compare_aucs(trueY1,scoreY1,trueY2,scoreY2, method='delong'):
    proc = importr('pROC')
    loc1 = trueY1.isnull()
    loc2 = scoreY1.isnull()
    locs =np.logical_not(np.logical_or(loc1,loc2))
    trueY1 = trueY1[locs]
    scoreY1 = scoreY1[locs]
    loc1 = trueY2.isnull()
    loc2 = scoreY2.isnull()
    locs =np.logical_not(np.logical_or(loc1,loc2))
    trueY2 = trueY2[locs]
    scoreY2 = scoreY2[locs]
    
    ground1 = ro.vectors.IntVector(trueY1)
    score1 = ro.vectors.FloatVector(scoreY1)
    ground2 = ro.vectors.IntVector(trueY2)
    score2 = ro.vectors.FloatVector(scoreY2)
    roc1 = proc.roc(ground1, score1,direction='<', ci=False)
    roc2 = proc.roc(ground2, score2, direction='<', ci=False)
    res = proc.roc_test(roc1, roc2, method=method)
    
    p = res.rx2('p.value')[0]
    return p

def report_performance(trueY, scoreY, n=None):
    # auc, 95%ci, tpr,tnr,fpr,fnr,ppv,npv,f1, odds ratio, OR 95%ci, P, N, tp, tn, fp, fn
    loc1 = trueY.isnull()
    loc2 = scoreY.isnull()
    locs =np.logical_not(np.logical_or(loc1,loc2))
    trueY = trueY[locs]
    scoreY = scoreY[locs]
    res = {}
    proc = importr('pROC')
    if n is None:
        ground = ro.vectors.IntVector(trueY)
        score = ro.vectors.FloatVector(scoreY)#[:,1])
    else:
        ground = ro.r.matrix(ro.IntVector(trueY), ncol=n)
        score = ro.r.matrix(ro.FloatVector(scoreY), ncol=n)#[:,1])
    roc1 = proc.roc(ground, score,direction='<', ci='True')  #roc1.names
    res['auc'] = roc1.rx2('ci')[1] # auroc
    res['auc_cilow'] = roc1.rx2('ci')[0] # low ci
    res['auc_cihigh'] = roc1.rx2('ci')[2] # high ci
    
    rocr= importr('ROCR')
    pre = rocr.prediction(score, ground)
    pref = rocr.performance(pre, 'sens', 'spec') #tuple(pref.slotnames())
    #print(np.array(pref.slots['x.values'][0]))
    sumsenspe = np.array(pref.slots['x.values'][0]) + np.array(pref.slots['y.values'][0])
    #print(sumsenspe)
    maxloc = np.argmax(sumsenspe)
    #print(pref.slots['y.values'][0])
    res_array = np.array(pref.slots['alpha.values'][0])
    tpr = np.array(pref.slots['y.values'][0])[maxloc]#)[0][maxloc]
    tnr = np.array(pref.slots['x.values'][0])[maxloc]
    res_array = np.vstack((res_array, np.array(pref.slots['y.values'][0])))
    res_array = np.vstack((res_array, np.array(pref.slots['x.values'][0])))
    fpr = 1 - tnr
    fnr = 1 - tpr
    res['cutoff'] = np.array(pref.slots['alpha.values'][0])[maxloc]
    res['tpr'] = tpr
    res['tnr'] = tnr
    res['fpr'] = fpr
    res['fnr'] = fnr
    pref = rocr.performance(pre, 'ppv')
    res['ppv'] = np.array(pref.slots['y.values'][0])[maxloc]
    res_array = np.vstack((res_array, np.array(pref.slots['y.values'][0])))
    pref = rocr.performance(pre, 'npv')
    res['npv'] = np.array(pref.slots['y.values'][0])[maxloc]
    res_array = np.vstack((res_array, np.array(pref.slots['y.values'][0])))
    pref = rocr.performance(pre, 'f')
    res['fscore'] = np.array(pref.slots['y.values'][0])[maxloc]
    res_array = np.vstack((res_array, np.array(pref.slots['y.values'][0])))
    pref = rocr.performance(pre, 'odds')
    res['odds'] = np.array(pref.slots['y.values'][0])[maxloc]
    res_array = np.vstack((res_array, np.array(pref.slots['y.values'][0]))).T
    res_array = pd.DataFrame(res_array, columns = ['score','TPR','TNR','PPV','NPV','F','ODDS'])
    P = np.sum(trueY)
    N = len(trueY) - P
    tp = tpr * P
    tn = tnr * N
    fp = fpr * N
    fn = fnr * P
    res['P'] = P
    res['N'] = N
    siglog = np.sqrt(1/tp + 1/tn + 1/fp + 1/fn)
    zalph = norm.ppf(0.975)
    #odds = tp*tn / (fp*fn)
    #print(odds)
    logOR = np.log(res['odds'])
    loglo = logOR - zalph * siglog
    loghi = logOR + zalph * siglog
    ORlo = np.exp(loglo)
    ORhi = np.exp(loghi)
    res['ORlo'] = ORlo
    res['ORhi'] = ORhi
    return res, res_array

def shap_analysis(model, data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    shap.summary_plot(shap_values, data, show=False)
    