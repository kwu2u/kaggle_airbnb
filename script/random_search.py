# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:26:12 2015

@author: kwu
"""

import xgboost as xgb
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
param = {'eta': 0.15, 'max_depth': 6,'subsample': 0.5, 'colsample_bytree': 0.5, 
         'objective': 'multi:softprob', 'num_class': 12}
num_round = 45
label2num = {label: i for i, label in enumerate(sorted(set(y)))}
dtrain = xgb.DMatrix(X, label=[label2num[label] for label in y])
xgb.cv(param, dtrain, num_round, nfold=3, metrics={'ndcg'}, seed = 0)
'''