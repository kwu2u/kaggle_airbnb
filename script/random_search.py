# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:26:12 2015

@author: kwu
"""
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
import holidays
import xgboost as xgb
from sklearn.grid_search import RandomizedSearchCV 

# Loading train, test, and sessions data
df_train = pd.read_csv('../input/train_users_2.csv')
df_test = pd.read_csv('../input/test_users.csv')
sessions = pd.read_csv('../input/sessions.csv')
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
train_test_cutoff = df_train.shape[0]

# Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

##### Aggregating sessions data (based on Abeer Jha post) #####
# only keeping sessions data with user ids found in train & test data
id_all = df_all['id']
sessions_rel = sessions[sessions.user_id.isin(id_all)]

# calculating total time elapsed
grp_by_sec_elapsed = sessions_rel.groupby(['user_id'])['secs_elapsed'].sum().reset_index()
grp_by_sec_elapsed.columns = ['user_id', 'secs_elapsed']

# aggregating by action and counting action_details
action = pd.pivot_table(sessions_rel, index = ['user_id'],
                        columns = ['action'],
                        values = 'action_detail',
                        aggfunc=len,fill_value=0).reset_index()
                        
# aggregating by action_type and counting actions                        
action_type = pd.pivot_table(sessions_rel, index = ['user_id'],
                             columns = ['action_type'],
                             values = 'action',
                             aggfunc=len,fill_value=0).reset_index()
                             
# aggregating by device_type and counting actions                             
device_type = pd.pivot_table(sessions_rel, index = ['user_id'],
                             columns = ['device_type'],
                             values = 'action',
                             aggfunc=len,fill_value=0).reset_index()
                             
# adding aggregated session features to dataframe                             
sessions_data = pd.merge(action_type, device_type, 
                         on='user_id', how='inner')
sessions_data = pd.merge(sessions_data, action, 
                         on='user_id',how='inner')
sessions_data = pd.merge(sessions_data, grp_by_sec_elapsed,
                         on='user_id', how='inner')
df_all = pd.merge(df_all, sessions_data, left_on='id', 
                  right_on='user_id', how='left')

# Removing id and date_first_booking
df_all = df_all.drop(['id', 'user_id', 'date_first_booking'], axis=1)
# Filling all nan with -1 
# tried imputing age with a rf, but did not improve results
df_all = df_all.fillna(-1)

##### Feature engineering #####
# creating a list of 10d windows before 5 major US holidays
holidays_tuples = holidays.US(years=[2010,2011,2012,2013,2014])
popular_holidays = ['Thanksgiving', 'Christmas Day', 'Independence Day', 
                    'Labor Day', 'Memorial Day']
holidays_tuples = {k:v for (k,v) in holidays_tuples.items() if v in popular_holidays}
us_holidays = pd.to_datetime([i[0] for i in np.array(holidays_tuples.items())])

def make_window(start, end, holiday_list):
    temp = [pd.date_range(j,i) for i,j in zip(holiday_list + pd.DateOffset(start),
            holiday_list + pd.DateOffset(end))]
    temp = holiday_list[len(holiday_list):].append(temp)
    return temp.unique()

holiday_30 = make_window(0, -30, us_holidays)

# date_account_created
dac = np.vstack(df_all.date_account_created.astype(str).apply(
    lambda x: list(map(int, x.split('-')))).values)
df_all.update({
    'dac_year': dac[:,0],
    'dac_month': dac[:,1],
    'dac_day': dac[:,2]
})
df_all['date_account_created'] = pd.to_datetime(df_all['date_account_created'])
df_all['dac_holiday_30'] = df_all.date_account_created.isin(holiday_30)

dac_day_of_wk = []
for date in df_all.date_account_created:
    dac_day_of_wk.append(date.weekday())
df_all['dac_day_of_wk'] = pd.Series(dac_day_of_wk)

df_all = df_all.drop(['date_account_created'], axis=1)

# timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(
            lambda x: list(map(int, [x[:4], x[4:6], x[6:8], 
                                     x[8:10], x[10:12], x[12:14]]))
        ).values)  
df_all.update({
    'tfa_year': tfa[:,0],
    'tfa_month': tfa[:,1],
    'tfa_day': tfa[:,2]
})
df_all['date_first_active'] = pd.to_datetime(
    (df_all.timestamp_first_active // 1000000), format='%Y%m%d')
df_all['tfa_holiday_30'] = df_all.date_first_active.isin(holiday_30)

tfa_day_of_wk = []
for date in df_all.date_first_active:
    tfa_day_of_wk.append(date.weekday())
df_all['tfa_day_of_wk'] = pd.Series(tfa_day_of_wk)

df_all = df_all.drop(['timestamp_first_active','date_first_active'], axis=1)

# Age 
# valid range (14-100), calculate birth date (1919-1995), else -1
av = df_all.age.values
df_all['age'] = np.where(np.logical_and(av>1919, av<1995), 2015-av, av)
df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

# One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 
             'affiliate_channel', 'affiliate_provider', 
             'first_affiliate_tracked', 'signup_app', 
             'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

   
#using feature selection done during CV
feat_keep = pd.read_csv('features.csv')
df_all = df_all[feat_keep.feature.values]


# Splitting train and test
vals = df_all.values
X = vals[:train_test_cutoff]
le = LabelEncoder()
y = le.fit_transform(labels)   
X_test = vals[train_test_cutoff:]

# Grid search hyperparameter tuning
clf = xgb.sklearn.XGBClassifier()

parameters = {
        'max_depth':[6], 
        'learning_rate':[0.1, 0.2, 0.3],
        'n_estimators':[30, 45, 60],
        'objective':'multi:softprob',
        'nthread':[-1],
        'gamma':[0],
        'min_child_weight':[1],
        'max_delta_step':[0],
        'subsample':[0.5],
        'colsample_bytree':[0.5],
        'base_score':[0.08333333, 0.5], 
        'seed':[0],     
}

fit_params = {'eval_metric':'ndcg'}

rand = RandomizedSearchCV(clf, parameters, cv=3, n_iter=15, 
                          fit_params=fit_params, verbose=1)
rand.fit(X,y)
scores =  rand.grid_scores_
print rand.best_score_
print rand.best_params_

'''
param = {'eta': 0.15, 'max_depth': 6,'subsample': 0.5, 'colsample_bytree': 0.5, 
         'objective': 'multi:softprob', 'num_class': 12}
num_round = 45
label2num = {label: i for i, label in enumerate(sorted(set(y)))}
dtrain = xgb.DMatrix(X, label=[label2num[label] for label in y])
xgb.cv(param, dtrain, num_round, nfold=3, metrics={'ndcg'}, seed = 0)
'''