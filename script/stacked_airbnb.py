from Munger import clean_data
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import scale

M = clean_data()
M.df_all[M.session_cols] = M.df_all[M.session_cols].apply(lambda x: scale(x))
cont_cols = ['age','dac_day','tfa_day']
M.df_all[cont_cols] = (M.df_all[cont_cols] - M.df_all[cont_cols].mean())/M.df_all[cont_cols].std()

X, X_test = M.data_split()
y = M.label_transformer()

# another classifier

feat_keep = pd.read_csv('features-kwu2u.csv')
M.select_features(feat_keep)

X, X_test = M.data_split()

X_stacked = np.concatenate((X, svc_train), axis=1)
X_test_stacked = np.concatenate((X_test, svc_test), axis=1)

opt_params = {'eta': 0.05, 
              'max_depth': 6,
              'subsample': 0.7, 
              'colsample_bytree': 0.7, 
              'objective': 'multi:softprob', 
              'num_class': 12,
              'eval_metric':'ndcg@5',
              'seed':1234}
              
label2num = {label: i for i, label in enumerate(sorted(set(y)))}
dtrain = xgb.DMatrix(X_stacked, label=[label2num[label] for label in y])
bst = xgb.train(params=opt_params, dtrain=dtrain, num_boost_round=200)

y_pred = bst.predict(xgb.DMatrix(X_test_stacked), 
                     ntree_limit=bst.best_iteration
                ).reshape(X_test.shape[0],12) 

# Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(M.id_test)):
    idx = M.id_test[i]
    ids += [idx] * 5
    cts += M.label_inverse_transformer(np.argsort(y_pred[i])[::-1])[:5].tolist()

# Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)