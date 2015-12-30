# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:26:12 2015

@author: kwu
"""
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('C:\\users\\kwu\\anaconda2\\lib\\site-packages\\xgboost-0.4-py2.7.egg')
import xgboost as xgb
from sklearn.grid_search import RandomizedSearchCV 

#Loading data
df_train = pd.read_csv('../input/train_users_2.csv')
df_test = pd.read_csv('../input/test_users.csv')
sessions = pd.read_csv('../input/sessions.csv')
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
train_test_cutoff = df_train.shape[0]

#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

#aggregating sessions data (based on Abeer Jha post) and adding to df_all
id_all = df_all['id']
sessions_rel = sessions[sessions.user_id.isin(id_all)]
grp_by_sec_elapsed = sessions_rel.groupby(['user_id'])['secs_elapsed'].sum().reset_index()
grp_by_sec_elapsed.columns = ['user_id','secs_elapsed'] #total time elapsed
action = pd.pivot_table(sessions_rel, index = ['user_id'],columns = ['action'],values = 'action_detail',aggfunc=len,fill_value=0).reset_index()
action_type = pd.pivot_table(sessions_rel, index = ['user_id'],columns = ['action_type'],values = 'action',aggfunc=len,fill_value=0).reset_index()
device_type = pd.pivot_table(sessions_rel, index = ['user_id'],columns = ['device_type'],values = 'action',aggfunc=len,fill_value=0).reset_index()
sessions_data = pd.merge(action_type,device_type,on='user_id',how='inner')
sessions_data = pd.merge(sessions_data,action,on='user_id',how='inner')
sessions_data = pd.merge(sessions_data,grp_by_sec_elapsed,on='user_id',how='inner')
df_all = pd.merge(df_all,sessions_data,left_on='id',right_on='user_id',how='left')

#Removing id and date_first_booking
df_all = df_all.drop(['id', 'user_id', 'date_first_booking'], axis=1)
#Filling nan
df_all = df_all.fillna(-1)

#####Feature engineering#######
#date_account_created
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
df_all = df_all.drop(['date_account_created'], axis=1)

#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)

#Age 
#valid range (14-100), calculate birth date (1919-1995), else -1
av = df_all.age.values
df_all['age'] = np.where(np.logical_and(av>1919, av<1995), 2015-av, av)
df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

#Splitting train and test
vals = df_all.values
X = vals[:train_test_cutoff]
le = LabelEncoder()
y = le.fit_transform(labels)   
X_test = vals[train_test_cutoff:]

#Grid search hyperparameter tuning
class XGBoostClassifier():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})
 
    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)
 
    def predict(self, X):
        num2label = {i: label for label, i in self.label2num.items()}
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])
 
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)
 
    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)
 
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self
    
    
def logloss(y_true, Y_pred):
    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in zip(Y_pred, y_true)) / len(Y_pred)


clf = XGBoostClassifier(
    eval_metric = 'ndcg',
    num_class = 12,
    silent = 1,
    )
parameters = {
    'num_boost_round': [25, 30, 35, 40, 45, 50],
    'eta': [0.05, 0.10, 0.15, 0.2, 0.25, 0.3],
    'max_depth': [6, 7, 8, 9, 10, 11, 12],
    'subsample': [0.5],
    'colsample_bytree': [0.5],
}

rand = RandomizedSearchCV(clf, parameters, cv=3, n_iter=30)
rand.fit(X,y)
print rand.grid_scores_
print rand.best_score_
print rand.best_params_


#using optimized model to do feature selection
opt_params = {'eta': 0.15, 'max_depth': 6,'subsample': 0.5, 'colsample_bytree': 0.5}
label2num = {label: i for i, label in enumerate(sorted(set(y)))}
dtrain = xgb.DMatrix(X, label=[label2num[label] for label in y])
bst = xgb.train(params=opt_params, dtrain=dtrain, num_boost_round=45)
xgb.plot_importance(bst)

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()

create_feature_map(list(df_all.columns.values))
importance = bst.get_fscore(fmap='xgb.fmap')
importance_df = pd.DataFrame(importance.items(), columns=['feature','fscore'])
importance_df.to_csv('features.csv',index=False)

#test=pd.read_csv('features.csv')
#df_all_trim_feat = df_all[test.feature.values]

'''
mean: 0.93932, std: 0.09906, params: {'subsample': 0.5, 'num_boost_round': 45, 'eta': 0.25, 'colsample_bytree': 0.5, 'max_depth': 6}
mean: 0.94178, std: 0.10318, params: {'subsample': 0.5, 'num_boost_round': 50, 'eta': 0.1, 'colsample_bytree': 0.5, 'max_depth': 6}
mean: 0.93384, std: 0.09553, params: {'subsample': 0.5, 'num_boost_round': 40, 'eta': 0.15, 'colsample_bytree': 0.5, 'max_depth': 11}
mean: 0.92916, std: 0.09324, params: {'subsample': 0.5, 'num_boost_round': 25, 'eta': 0.25, 'colsample_bytree': 0.5, 'max_depth': 10}
mean: 0.89720, std: 0.08399, params: {'subsample': 0.5, 'num_boost_round': 30, 'eta': 0.1, 'colsample_bytree': 0.5, 'max_depth': 10}
mean: 0.89385, std: 0.08079, params: {'subsample': 0.5, 'num_boost_round': 50, 'eta': 0.3, 'colsample_bytree': 0.5, 'max_depth': 12}
mean: 0.92253, std: 0.09070, params: {'subsample': 0.5, 'num_boost_round': 40, 'eta': 0.25, 'colsample_bytree': 0.5, 'max_depth': 10}
mean: 0.93159, std: 0.09586, params: {'subsample': 0.5, 'num_boost_round': 30, 'eta': 0.15, 'colsample_bytree': 0.5, 'max_depth': 9}
mean: 0.83069, std: 0.06626, params: {'subsample': 0.5, 'num_boost_round': 40, 'eta': 0.05, 'colsample_bytree': 0.5, 'max_depth': 7}
mean: 0.93847, std: 0.10063, params: {'subsample': 0.5, 'num_boost_round': 30, 'eta': 0.3, 'colsample_bytree': 0.5, 'max_depth': 6}
mean: 0.83067, std: 0.06632, params: {'subsample': 0.5, 'num_boost_round': 40, 'eta': 0.05, 'colsample_bytree': 0.5, 'max_depth': 9}
mean: 0.91233, std: 0.08447, params: {'subsample': 0.5, 'num_boost_round': 35, 'eta': 0.3, 'colsample_bytree': 0.5, 'max_depth': 11}
mean: 0.72828, std: 0.04037, params: {'subsample': 0.5, 'num_boost_round': 25, 'eta': 0.05, 'colsample_bytree': 0.5, 'max_depth': 12}
mean: 0.94523, std: 0.10367, params: {'subsample': 0.5, 'num_boost_round': 40, 'eta': 0.15, 'colsample_bytree': 0.5, 'max_depth': 6}
mean: 0.93210, std: 0.09735, params: {'subsample': 0.5, 'num_boost_round': 45, 'eta': 0.1, 'colsample_bytree': 0.5, 'max_depth': 10}
mean: 0.93862, std: 0.09653, params: {'subsample': 0.5, 'num_boost_round': 25, 'eta': 0.3, 'colsample_bytree': 0.5, 'max_depth': 7}
## mean: 0.94693, std: 0.10458, params: {'subsample': 0.5, 'num_boost_round': 45, 'eta': 0.15, 'colsample_bytree': 0.5, 'max_depth': 6}
mean: 0.93920, std: 0.09947, params: {'subsample': 0.5, 'num_boost_round': 35, 'eta': 0.2, 'colsample_bytree': 0.5, 'max_depth': 8}
mean: 0.93490, std: 0.09907, params: {'subsample': 0.5, 'num_boost_round': 45, 'eta': 0.1, 'colsample_bytree': 0.5, 'max_depth': 8}
mean: 0.92864, std: 0.09258, params: {'subsample': 0.5, 'num_boost_round': 30, 'eta': 0.25, 'colsample_bytree': 0.5, 'max_depth': 10}
mean: 0.92117, std: 0.08788, params: {'subsample': 0.5, 'num_boost_round': 35, 'eta': 0.3, 'colsample_bytree': 0.5, 'max_depth': 9}
mean: 0.82679, std: 0.06248, params: {'subsample': 0.5, 'num_boost_round': 40, 'eta': 0.05, 'colsample_bytree': 0.5, 'max_depth': 12}
mean: 0.94130, std: 0.10202, params: {'subsample': 0.5, 'num_boost_round': 25, 'eta': 0.25, 'colsample_bytree': 0.5, 'max_depth': 7}
mean: 0.92248, std: 0.09218, params: {'subsample': 0.5, 'num_boost_round': 40, 'eta': 0.1, 'colsample_bytree': 0.5, 'max_depth': 12}
mean: 0.92441, std: 0.09163, params: {'subsample': 0.5, 'num_boost_round': 40, 'eta': 0.2, 'colsample_bytree': 0.5, 'max_depth': 12}
mean: 0.76952, std: 0.05042, params: {'subsample': 0.5, 'num_boost_round': 30, 'eta': 0.05, 'colsample_bytree': 0.5, 'max_depth': 10}
mean: 0.93566, std: 0.09776, params: {'subsample': 0.5, 'num_boost_round': 50, 'eta': 0.1, 'colsample_bytree': 0.5, 'max_depth': 11}
mean: 0.86992, std: 0.07476, params: {'subsample': 0.5, 'num_boost_round': 25, 'eta': 0.1, 'colsample_bytree': 0.5, 'max_depth': 11}
mean: 0.93406, std: 0.09635, params: {'subsample': 0.5, 'num_boost_round': 45, 'eta': 0.2, 'colsample_bytree': 0.5, 'max_depth': 9}
mean: 0.92700, std: 0.09141, params: {'subsample': 0.5, 'num_boost_round': 30, 'eta': 0.15, 'colsample_bytree': 0.5, 'max_depth': 11}
'''