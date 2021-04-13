# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 00:36:50 2020

@author: 14109
"""

import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold,RepeatedKFold,StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import shap
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import class_weight
import matplotlib.pyplot as plt 

# xgboost tree
# elastic net
# logistic regression



def trainXGB(trX, trY, teX=None, teY=None, perfEval=False):
    seed = 255
    estimator = XGBClassifier(objective= 'binary:logistic', booster='gbtree',nthread=4,seed=12)
    #parameters = {'max_depth': range (2, 10, 1),'n_estimators': range(60, 220, 40),
    #              'learning_rate': [0.1, 0.01, 0.05],'colsample_bytree':uniform(0.7, 0.3),
    #              'gamma':uniform(0,0.5),'subsample':uniform(0.6,0.4)}
    #rkf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=seed)
    rkf = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
    #grid_search = GridSearchCV(estimator=estimator,param_grid=parameters,
    #                           scoring = 'roc_auc',n_jobs = -1,cv = rkf,verbose=False)
    #grid_search.fit(X, Y)
    #grid_search.best_estimator_
    parameters = {'max_depth': randint(2, 3),'n_estimators': randint(70, 100),
                  'learning_rate': uniform(0.015, 0.06),'min_child_weight':uniform(10,5),'colsample_bytree':uniform(0.85, 0.15),
                  'gamma':uniform(3,10)}#,'subsample':uniform(0.85,0.15)}
    rand_search = RandomizedSearchCV(estimator=estimator,param_distributions=parameters,
                                     random_state=seed,n_iter=15,cv=rkf,n_jobs=-1,scoring='roc_auc_ovr',verbose=True)
    
    class_weights = list(class_weight.compute_class_weight('balanced',np.unique(trY),trY))
    sample_weight = np.ones(trY.shape[0], dtype = 'float')
    for i, val in enumerate(trY):
        sample_weight[i] = class_weights[int(val)]
    rand_search.fit(trX, trY, sample_weight = sample_weight)
    
    bstModel = rand_search.best_estimator_
    # Save model
    #pickle.dump(bstModel, open("/home/kalman/temp/xgb_demo_wv_withISS.pickle", "wb"))
    
    #report_best_scores(rand_search.cv_results_, 1)
    #xgb.plot_importance(bstModel)
    #score = bstModel.predict_proba(teX)
    #scoretr = bstModel.predict_proba(trX)
    #roc_auc_score(teY, score[:,1])
    #fpr, tpr, thresholds = roc_curve(teY, score[:,1])
    #plot_roc_curve(bstModel, teX, teY)
    if perfEval:
        scoreTr = bstModel.predict_proba(trX)
        scoreTe = bstModel.predict_proba(teX)
        #explainer = shap.TreeExplainer(bstModel)
        #shapValues = explainer.shap_values(trX)
        #shapSummary = shap.summary_plot(shapValues, trX, show=False)
    else:
        scoreTr = None
        scoreTe = None
        #shapValues = None
        #shapSummary = None
    #model_exp5 = rand_search.best_estimator_
    
    return bstModel, scoreTr, scoreTe#, trY, teY#, shapValues, shapSummary


def trainLogisticRegressionElasticNet(trX, trY, teX=None, teY=None, perfEval=False):
    seed = 4242341
    trX = trX.dropna()
    trY = trY.loc[trX.index]
    teX = teX.dropna()
    teY = teY.loc[teX.index]
    scaler = StandardScaler()
    scaler.fit(trX)
    trX = scaler.transform(trX)
    teX = scaler.transform(teX)
    clf = LogisticRegressionCV(cv=5, random_state=seed,penalty='elasticnet',solver='saga',scoring='roc_auc',l1_ratios=[0.1,0.3,0.5,0.7,0.9]).fit(trX, trY)
    scoreTr = clf.predict_proba(trX)
    #roc_auc_score(trY, scoreTr[:,1])
    scoreTe = clf.predict_proba(teX)
    #roc_auc_score(teY, scoreTe[:,1])
    
    return scoreTr, trY, scoreTe, teY



# prepare a dataframe
# stratified CV, repeated
# collect each train, test result

# for binary, multi-class, continuous outcomes

file = r'P:\_Projects\ONPOINT3\DrFloccare\Study2-Trauma triage threshold vs transfusion requirement\data14437.xlsx'
dataAll = pd.read_excel(file, 0)
dataAll['si_pre'] = dataAll['hr_pre'] / dataAll['sysbp_pre']
dataAll['si_tru'] = dataAll['hr_tru_adm'] / dataAll['sysbp_tru_adm']

outcomes = ['UnX','CAT','MT']

featurecols1 = {}
featurecols1[0] = ['sysbp_pre','hr_pre']
featurecols1[1] = ['sysbp_pre','hr_pre','isPenetrating']
featurecols1[2] = ['si_pre']
featurecols1[3] = ['si_pre','isPenetrating']
featurecols1[4] = ['sysbp_tru_adm','hr_tru_adm']
featurecols1[5] = ['sysbp_tru_adm','hr_tru_adm', 'isPenetrating']
featurecols1[6] = ['si_tru']
featurecols1[7] = ['si_tru','isPenetrating']

counter = 0

for outcome in outcomes:
    
    for exp in range(8):
                
        traindata = dataAll[featurecols1[exp]+[outcome]]
        traindata[outcome] = traindata[outcome].astype('int')
        traindata.dropna(inplace=True)

        n_split = 5
        seeds = [231,544,323,1341,24,323,123,56,458,456,678,245,756,41,8,94252,446,78,5624,37]
        
        #res_xgb = pd.DataFrame( np.zeros((n_case, n_split*1+1 )) )
        #res_xgb[0] = traindata['decline']
        roc_xgb = pd.DataFrame( np.zeros((2, n_split*10)) )
        cnt = 1
        trlen = round(traindata.shape[0] * 0.8)+1
        telen = round(traindata.shape[0] * 0.2)+1
        trY = np.empty([trlen, 50])
        trY[:] = np.NaN
        AllscoreTr = np.empty([trlen, 50])
        AllscoreTr[:] = np.NaN
        teY = np.empty([telen, 50])
        teY[:] = np.NaN
        AllscoreTe = np.empty([telen, 50])
        AllscoreTr[:] = np.NaN
        fig1, ax1 = plt.subplots()
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        #featurecols = list(compress(featurecols1, flocs)) + [546,547,436,468,471,518,549]
        for i in range(10):
        
            #X_train, X_test, y_train, y_test = train_test_split(traindata, traindata['decline'], test_size=0.33, random_state=4231, stratify=traindata['decline'])
            skf = StratifiedKFold(n_splits=n_split,shuffle=True,random_state=seeds[i])
            y = traindata[outcome].astype(int)
        
            for train_index, test_index in skf.split(traindata, y):
                X_train, X_test = traindata.iloc[train_index][featurecols1[exp]], traindata.iloc[test_index][featurecols1[exp]]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                
                #bstModel,scoreTr,scoreTe = trainXGB(X_train.iloc[:,featurecols], y_train, X_test.iloc[:,featurecols],y_test,True)
                bstModel = LogisticRegression(random_state=0).fit(X_train, y_train)
                scoreTr = bstModel.predict_proba(X_train)
                scoreTe = bstModel.predict_proba(X_test)
                res_obj = [bstModel, scoreTr, scoreTe, y_train, y_test, train_index,test_index]
                with open('C:/projects/NW/results/exp'+str(exp)+'_10foldLR_OP3_'+outcome+'_14437.dat','wb') as nwf:
                    pickle.dump(res_obj, nwf)
                #res_xgb[cnt].iloc[train_index] = scoreTr[:,1]+1
                #res_xgb[cnt].iloc[test_index] = scoreTe[:,1]
                #mcoutcomeTr = LabelBinarizer().fit_transform(y_train)
                roc_xgb.iloc[0,cnt-1] = roc_auc_score(y_train, scoreTr[:,1])
                roc_xgb.iloc[1,cnt-1] = roc_auc_score(y_test, scoreTe[:,1])
                #roc_xgb.iloc[2,cnt-1] = roc_auc_score(y6_test, scoreTe[:,1])
                #roc_xgb.iloc[1,cnt-1] = roc_auc_score(y12_test, scoreTe[:,1])
                cnt = cnt + 1
                
                
                trY[0:len(y_train), cnt] = y_train  # trY + thisTrY#.values.tolist()
                teY[0:len(y_test), cnt] = y_test  # teY + thisTeY#.values.tolist()
                AllscoreTr[0:len(scoreTr[:,1]), cnt] = scoreTr[:,1]  # scoreTr + thisScoreTr#.values.tolist()list()
                AllscoreTe[0:len(scoreTe), cnt] = scoreTe[:,1]  # scoreTe + thisScoreTe#.values.tolist()
                perf.plot_roc_curve_r(pd.DataFrame(y_train)[0], pd.DataFrame(scoreTr[:,1])[0], color='#FFD7E6', label=None,
                                      fig=fig1, ax=ax1, isLast=False)
                perf.plot_roc_curve_r(pd.DataFrame(y_test)[0], pd.DataFrame(scoreTe[:,1])[0], color='#ADDFFF', label=None,
                                      fig=fig1, ax=ax1, isLast=False)
                
                
        trY = pd.DataFrame(trY.flatten('F'))[0]
        AllscoreTr = pd.DataFrame(AllscoreTr.flatten('F'))[0]
        teY = pd.DataFrame(teY.flatten('F'))[0]
        AllscoreTe = pd.DataFrame(AllscoreTe.flatten('F'))[0]
        perf.plot_roc_curve_r(trY, AllscoreTr, color='#9F000F', label='Train Average', fig=fig1, ax=ax1, isLast=False)
        perf.plot_roc_curve_r(teY, AllscoreTe, color='#2B65EC', label='Test Average ', fig=fig1, ax=ax1, isLast=True)
        plt.saveFig('C:/projects/tmp/op3_14437_'+outcome+'_exp'+str(exp)+'.png')
        
        resTR[counter,:], _ = perf.report_performance(trY, AllscoreTr, 50)
        resTE[counter,:], _ = perf.report_performance(teY, AllscoreTe, 50)
        counter = counter + 1

#import matplotlib.pyplot as plt
#from atool.Python_utils.MLCore import performance as perf  #performance as perf

trY = np.empty([1779, 100])
trY[:] = np.NaN
scoreTr = np.empty([1779, 100])
scoreTr[:] = np.NaN
teY = np.empty([446, 100])
teY[:] = np.NaN
scoreTe = np.empty([446, 100])
scoreTr[:] = np.NaN

fig1, ax1 = plt.subplots()
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
# fig2, ax2 = plt.subplots()
for cnt in range(50):
    with open('C:/projects/NW/results/exp' + str(cnt + 1) + '_10foldLR_OP3_'+outcome+'_t4.dat', 'rb') as nwf:
        unpickler = pickle.Unpickler(nwf)
        obj = unpickler.load()
        thisScoreTr = list(obj[1][:, 1])
        thisScoreTe = list(obj[2][:, 1])
        thisTrY = list(obj[3])
        thisTeY = list(obj[4])

        perfres, _ = perf.report_performance(pd.DataFrame(thisTeY)[0], pd.DataFrame(thisScoreTe)[0])
        if perfres['auc'] > 0.0:
            trY[0:len(thisTrY), cnt] = thisTrY  # trY + thisTrY#.values.tolist()
            teY[0:len(thisTeY), cnt] = thisTeY  # teY + thisTeY#.values.tolist()
            scoreTr[0:len(thisScoreTr), cnt] = thisScoreTr  # scoreTr + thisScoreTr#.values.tolist()list()
            scoreTe[0:len(thisScoreTe), cnt] = thisScoreTe  # scoreTe + thisScoreTe#.values.tolist()
            perf.plot_roc_curve_r(pd.DataFrame(thisTrY)[0], pd.DataFrame(thisScoreTr)[0], color='#FFD7E6', label=None,
                                  fig=fig1, ax=ax1, isLast=False)
            perf.plot_roc_curve_r(pd.DataFrame(thisTeY)[0], pd.DataFrame(thisScoreTe)[0], color='#ADDFFF', label=None,
                                  fig=fig1, ax=ax1, isLast=False)

trY = pd.DataFrame(trY.flatten('F'))[0]
scoreTr = pd.DataFrame(scoreTr.flatten('F'))[0]
teY = pd.DataFrame(teY.flatten('F'))[0]
scoreTe = pd.DataFrame(scoreTe.flatten('F'))[0]
perf.plot_roc_curve_r(trY, scoreTr, color='#9F000F', label='Train Average', fig=fig1, ax=ax1, isLast=False)
perf.plot_roc_curve_r(teY, scoreTe, color='#2B65EC', label='Test Average ', fig=fig1, ax=ax1, isLast=True)

resTR, resTRall = perf.report_performance(trY, scoreTr, 50)
resTE, resTEall = perf.report_performance(teY, scoreTe, 50)