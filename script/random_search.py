from Munger import clean_data
import pandas as pd
import numpy as np
import xgboost as xgb

M = clean_data()

feat_keep = pd.read_csv('features.csv')
M.select_features(feat_keep)

X, X_test = M.data_split()
y = M.label_transformer()

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
cv-test-ndcg5:0.832291666667 current best model
cv-test-ndcg5:0.832114666667 w/ binned age
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