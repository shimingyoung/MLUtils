# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 00:36:50 2020

@author: 14109
"""

import pandas as pd
import numpy as np
import pycaret
import pickle
from pycaret.classification import *


# xgboost tree
# elastic net
# logistic regression
file = r'P:\_Projects\ONPOINT3\DrFloccare\Study2-Trauma triage threshold vs transfusion requirement\data2223.xlsx'
dataAll = pd.read_excel(file, 0)
data = dataAll[['isPenetrating','sysbp_pre','hr_pre','MT']]
data['MT'] = data['MT'].astype('int')
#data['CAT'] = data['MT'].astype('int')
#data['UnX'] = data['UnX'].astype('int')


# prepare a dataframe
# stratified CV, repeated
# collect each train, test result

# for binary, multi-class, continuous outcomes

datatrain = data.sample(frac=0.75, random_state=786)
data_unseen = data.drop(datatrain.index)
datatrain.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)

clf1 = setup(data = datatrain, target = 'MT', normalize = True, normalize_method = 'robust')
lr = create_model('lr')
#tuned_lr = tune_model(lr)
plot_model(lr, plot = 'auc')
unseen_predictions = predict_model(lr, data=data_unseen)

n_split = 10
seeds = [231,544,323,1341,24,323,123,56,458,456,678,245,756,41,8,94252,446,78,5624,37]
#res_xgb = pd.DataFrame( np.zeros((n_case, n_split*1+1 )) )
#res_xgb[0] = traindata['decline']
roc_xgb = pd.DataFrame( np.zeros((4, n_split*10)) )
cnt = 1
featurecols1 = fcols + fcols_scores
featurecols = list(compress(featurecols1, flocs)) + [546,547,436,468,471,518,549]
for i in range(10):

    #X_train, X_test, y_train, y_test = train_test_split(traindata, traindata['decline'], test_size=0.33, random_state=4231, stratify=traindata['decline'])
    skf = StratifiedKFold(n_splits=n_split,shuffle=True,random_state=seeds[i])
    y = traindata['decline'].astype(int)
    y6 = traindata['NW6'].astype(int)
    y12 = traindata['NW12'].astype(int)
    for train_index, test_index in skf.split(traindata, y):
        X_train, X_test = pd.concat([traindata.iloc[train_index], earlyNWdata],axis=0), traindata.iloc[test_index]
        y_train, y_test = pd.concat([y.iloc[train_index], earlyNWdata['decline'].astype(int)],axis=0), y.iloc[test_index]
        y6_test = y6.iloc[test_index]
        y12_test = y12.iloc[test_index]
        #print(sum(y_train)/len(y_train))
        #print(sum(y_test)/len(y_test))
        
        bstModel,scoreTr,scoreTe = trainXGB(X_train.iloc[:,featurecols], y_train, X_test.iloc[:,featurecols],y_test,True)
        res_obj = [bstModel, scoreTr, scoreTe, y_train, y_test, y6_test, y12_test, train_index,test_index]
        with open('C:/projects/NW/results/exp'+str(cnt)+'_10foldwithDemoINJ_2.dat','wb') as nwf:
            pickle.dump(res_obj, nwf)
        #res_xgb[cnt].iloc[train_index] = scoreTr[:,1]+1
        #res_xgb[cnt].iloc[test_index] = scoreTe[:,1]
        #mcoutcomeTr = LabelBinarizer().fit_transform(y_train)
        roc_xgb.iloc[0,cnt-1] = roc_auc_score(y_train, scoreTr[:,1])
        roc_xgb.iloc[1,cnt-1] = roc_auc_score(y_test, scoreTe[:,1])
        roc_xgb.iloc[2,cnt-1] = roc_auc_score(y6_test, scoreTe[:,1])
        #roc_xgb.iloc[1,cnt-1] = roc_auc_score(y12_test, scoreTe[:,1])
        cnt = cnt + 1