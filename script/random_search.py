# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:26:12 2015

@author: kwu
"""

import numpy as np
import xgboost as xgb
'''
from sklearn.grid_search import RandomizedSearchCV 

# Grid search hyperparameter tuning
clf = xgb.sklearn.XGBClassifier()

parameters = {
        'max_depth':[6], 
        'learning_rate':[0.2],
        'n_estimators':[20],
        'objective':'multi:softprob',
        'nthread':[-1],
        'gamma':[0],
        'min_child_weight':[1],
        'max_delta_step':[0],
        'subsample':[0.5],
        'colsample_bytree':[0.7],
        'base_score':[0.5], 
        'seed':[0],     
}

fit_params = {'eval_metric':'ndcg'}

rand = RandomizedSearchCV(clf, parameters, cv=3, n_iter=1, 
                          fit_params=fit_params, verbose=1)
                          
# run preprocessing in kwu2u_airbnb.py                          
rand.fit(X,y)
scores =  rand.grid_scores_
print rand.best_score_
print rand.best_params_

# 0.608678338354
'''
# run preprocessing in kwu2u_airbnb.py   

xgb_params = {'eta': 0.05, 
              'max_depth': 6,
              'subsample': 0.7, 
              'colsample_bytree': 0.7, 
              'objective': 'multi:softprob', 
              'num_class': 12,
              'eval_metric':'ndcg',
              'seed':1234}
              
def customized_eval(preds, dtrain):
    labels = dtrain.get_label()
    top = []
    for i in range(preds.shape[0]):
        top.append(np.argsort(preds[i])[::-1][:5])
    mat = np.reshape(np.repeat(labels, 
                               np.shape(top)[1]) == np.array(top).ravel(),
                               np.array(top).shape
                               ).astype(int)
    score = np.mean(np.sum(mat/np.log2(np.arange(2, mat.shape[1] + 2)),axis = 1))
    return 'ndcg5', score

label2num = {label: i for i, label in enumerate(sorted(set(y)))}
dtrain = xgb.DMatrix(X, label=[label2num[label] for label in y])
cv = xgb.cv(xgb_params, dtrain, num_boost_round=200, nfold=3, seed=0, 
             feval = customized_eval, maximize = True, show_progress = True)

'''
cv-test-ndcg5:0.832291666667
xgb_params = {'eta': 0.05, 
              'max_depth': 6,
              'subsample': 0.7, 
              'colsample_bytree': 0.7, 
              'objective': 'multi:softprob', 
              'num_class': 12,
              'eval_metric':'ndcg',
              'seed':1234}
num_boost_round=200
nfold=3
'''