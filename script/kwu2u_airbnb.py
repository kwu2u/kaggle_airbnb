# This script is based on Sandro's python script

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
import holidays
sys.path.append('C:\\users\\kwu\\anaconda2\\lib\\site-packages\\xgboost-0.4-py2.7.egg')
import xgboost as xgb

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
action = pd.pivot_table(sessions_rel, index = ['user_id'],
                        columns = ['action'],
                        values = 'action_detail',
                        aggfunc=len,fill_value=0).reset_index()
action_type = pd.pivot_table(sessions_rel, index = ['user_id'],
                             columns = ['action_type'],
                             values = 'action',
                             aggfunc=len,fill_value=0).reset_index()
device_type = pd.pivot_table(sessions_rel, index = ['user_id'],
                             columns = ['device_type'],
                             values = 'action',
                             aggfunc=len,fill_value=0).reset_index()
sessions_data = pd.merge(action_type,device_type,on='user_id',how='inner')
sessions_data = pd.merge(sessions_data,action,on='user_id',how='inner')
sessions_data = pd.merge(sessions_data,grp_by_sec_elapsed,on='user_id',how='inner')
df_all = pd.merge(df_all,sessions_data,left_on='id',right_on='user_id',how='left')

#Removing id and date_first_booking
df_all = df_all.drop(['id', 'user_id', 'date_first_booking'], axis=1)
#Filling nan
df_all = df_all.fillna(-1)

#####Feature engineering#######
#US holidays
holidays_tuples = holidays.US(years=[2010,2011,2012,2013,2014])
popular_holidays = ['Thanksgiving', 'Christmas Day', 'Independence Day', 
                    'Labor Day', 'Memorial Day']
holidays_tuples = {k:v for (k,v) in holidays_tuples.items() if v in popular_holidays}
us_holidays = [i[0] for i in np.array(holidays_tuples.items())]
us_holidays = pd.to_datetime(us_holidays)
us_holidays30 = us_holidays + pd.DateOffset(-30)
us_holidays_window = [pd.date_range(j,i) for i,j in zip(us_holidays,us_holidays30)]
us_holidays_window = us_holidays.append(us_holidays_window)
us_holidays_window = us_holidays_window.unique()
#date_account_created
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
df_all['date_account_created'] = pd.to_datetime(df_all['date_account_created'])
df_all['dac_holiday'] = df_all.date_account_created.isin(us_holidays_window)
dac_day_of_wk = []
for date in df_all.date_account_created:
    dac_day_of_wk.append(date.weekday())
df_all['dac_day_of_wk'] = pd.Series(dac_day_of_wk)
df_all = df_all.drop(['date_account_created'], axis=1)

#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all['date_first_active'] = pd.to_datetime((df_all.timestamp_first_active // 1000000), format='%Y%m%d')
df_all['tfa_holiday'] = df_all.date_first_active.isin(us_holidays_window)
tfa_day_of_wk = []
for date in df_all.date_first_active:
    tfa_day_of_wk.append(date.weekday())
df_all['tfa_day_of_wk'] = pd.Series(tfa_day_of_wk)
df_all = df_all.drop(['timestamp_first_active','date_first_active'], axis=1)

#Age 
#valid range (14-100), calculate birth date (1919-1995), else -1
av = df_all.age.values
df_all['age'] = np.where(np.logical_and(av>1919, av<1995), 2015-av, av)
df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 
             'affiliate_channel', 'affiliate_provider', 
             'first_affiliate_tracked', 'signup_app', 
             'first_device_type', 'first_browser']
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

#Classifier
opt_params = {'eta': 0.2, 'max_depth': 6,'subsample': 0.5, 
              'colsample_bytree': 0.5, 'objective': 'multi:softprob', 
              'num_class': 12}
label2num = {label: i for i, label in enumerate(sorted(set(y)))}
dtrain = xgb.DMatrix(X, label=[label2num[label] for label in y])
bst = xgb.train(params=opt_params, dtrain=dtrain, num_boost_round=45)

y_pred = bst.predict(xgb.DMatrix(X_test)).reshape(df_test.shape[0],12) 

#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)