# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 23:42:57 2020

@author: 14109
"""
import atool.Python_utils.MLCore.performance as perf
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def featureset_grouptest(data, ylabel, colidx, fname=None,sheet=None):
    cols = data.columns
    loc_pos = data[ylabel] == 1
    loc_neg = data[ylabel] == 0
    res = []
    res2 = []
    res_mean1 = []
    res_mean2 = []
    res_std1 = []
    res_std2 = []
    
    #colidx = featurecols1  #+ list(np.arange())#list(np.arange(3,283))+list(np.arange(311,323))
    for i in colidx:#(89,366):
        tmp = stats.ttest_ind( data.iloc[:,i].loc[loc_pos].dropna(), data.iloc[:,i].loc[loc_neg].dropna() )
        res.append(tmp.pvalue)
        try:
            tmp = stats.mannwhitneyu( data.iloc[:,i].loc[loc_pos].dropna(), data.iloc[:,i].loc[loc_neg].dropna() )
            res2.append(tmp.pvalue)
        except:
            res2.append(1.0)
        
        res_mean1.append(np.round(np.mean( data.iloc[:,i].loc[loc_pos].dropna() ),3 ))
        res_std1.append(np.round(np.std( data.iloc[:,i].loc[loc_pos].dropna() ),3 ))
        res_mean2.append(np.round(np.mean( data.iloc[:,i].loc[loc_neg].dropna() ),3 ))
        res_std2.append(np.round(np.std( data.iloc[:,i].loc[loc_neg].dropna() ),3 ))
        
    #a = pd.Series(res, index=cols[colidx])#(89,366)])
    b = pd.DataFrame(list(zip(res,res2,res_mean1,res_std1,res_mean2,res_std2)), columns = ['p ttest','p ranksum','mean1','std1','mean2','std2'])
    b.index=cols[colidx]#(89,366)])
    #res = pd.DataFrame(res1.items())
    #for i in colidx:
    #    res1 = perf.report_performance(traindata['decline'], traindata[cols[i]])
    #    res[cols[i]] = res1.values()                                                               
    if fname is not None:
        with pd.ExcelWriter(fname, engine = 'xlsxwriter') as writer:
            b.to_excel(writer,sheet_name=sheet)
            
            
def featureset_uniroc(data, ylabel, colidx, fname=None):
    colname = data.columns
    colidxname = colname[colidx]
    res = []
    for i in colidx:
        try:
            metric, _ = perf.report_performance(data[ylabel], data.iloc[:,i])
            if metric['auc'] < 0.5:
                metric, _ = perf.report_performance(data[ylabel], -1*data.iloc[:,i])
            res.append(list(metric.values()))
        except:
            res.append(list(metric.values()))
    res = pd.DataFrame(res, index=colidxname, columns=list(metric.keys()))
    
    if fname is not None:
        with pd.ExcelWriter(fname, engine = 'xlsxwriter') as writer:
            res.to_excel(writer,'sheet1')
    
    return res
    

def featureset_boxplot(data, ylabel, perfdata, by, nrow, ncol):
    sorted_df = perfdata.sort_values(by=by,ascending=False)
    #fig, axs = plt.subplots(nrows=nrow, ncols=ncol)
    nlocs = data[ylabel] == 0
    plocs = data[ylabel] == 1
    colors = ['#3BB9FF','#9F000F']
    for i in range(nrow*ncol):
        vname = sorted_df.iloc[i,0]
        plt.subplot(nrow,ncol, i+1)
        ndata = data[vname][nlocs].dropna()
        pdata = data[vname][plocs].dropna()
        thisdata = [ndata, pdata]
        bplot = plt.boxplot(thisdata, notch=True,sym='',patch_artist=True)
        plt.ylabel(vname,fontsize=16)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        
        
 