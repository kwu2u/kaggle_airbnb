from Munger import clean_data
from xgb_sklearn_wrapper import XGBoostClassifier
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier

M = clean_data()

feat_keep = pd.read_csv('features-kwu2u.csv')
M.select_features(feat_keep)

X, X_test = M.data_split()
y = M.label_transformer()

clf = XGBoostClassifier(
    eval_metric = 'ndcg@5',
    num_class = 12,
    silent = 1,
    )
    
clf.set_params(**{'num_boost_round': 200,
                  'eta' : 0.05,
                  'max_depth' : 6,
                  'subsample' : 0.7,
                  'colsample_bytree' : 0.7})
    
ovrClf = OneVsRestClassifier(clf)
ovrClf.fit(X, y)
y_pred = ovrClf.predict_proba(X_test)

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