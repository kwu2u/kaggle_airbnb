# This script is based on Sandro's python script

import numpy as np
import pandas as pd
import holidays
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('C:\\users\\kwu\\anaconda2\\lib\\site-packages\\xgboost-0.4-py2.7.egg')
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestRegressor

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
df_all['date_account_created'] = pd.to_datetime(df_all['date_account_created'])
#dac_day_of_wk = []
#for date in df_all.date_account_created:
#    dac_day_of_wk.append(date.weekday())
#df_all['dac_day_of_wk'] = pd.Series(dac_day_of_wk)
df_all = df_all.drop(['date_account_created'], axis=1)

#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all['date_first_active'] = pd.to_datetime((df_all.timestamp_first_active // 1000000), format='%Y%m%d')
#tfa_day_of_wk = []
#for date in df_all.date_first_active:
#    tfa_day_of_wk.append(date.weekday())
#df_all['tfa_day_of_wk'] = pd.Series(tfa_day_of_wk)
df_all = df_all.drop(['timestamp_first_active','date_first_active'], axis=1)

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
    
#missing age imputation
def setMissingAges(df):
    # Split into sets with known and unknown Age values
    knownAge = df.loc[ (df['age'] != -1) ]
    unknownAge = df.loc[ (df['age'] == -1) ]
    # All age values are stored in a target array
    y = knownAge.values[:,0]
    # All the other values are stored in the feature array
    X = knownAge.values[:, 1::]
    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=20, n_jobs=-1)
    rtr.fit(X, y)
    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(unknownAge.values[:,1::])
    # Assign those predictions to the full data set
    df.loc[ (df['age'] == -1), 'age' ] = predictedAges
    return df.age

df_all.age = setMissingAges(df_all)


#using feature selection done during CV
feat_keep = pd.read_csv('features.csv')
df_all = df_all[feat_keep.feature.values]


#Splitting train and test
vals = df_all.values
X = vals[:train_test_cutoff]
le = LabelEncoder()
y = le.fit_transform(labels)   
X_test = vals[train_test_cutoff:]

#Classifier
xgb = XGBClassifier(max_depth=6, learning_rate=0.2, n_estimators=45,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)                  
xgb.fit(X, y)
y_pred = xgb.predict_proba(X_test)  

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