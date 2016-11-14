# This script is based on Sandro's python script
# note: tree ensembles are invariant to scaling so no normalization necessary,
# one hot encoding categorical features

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import holidays
import xgboost as xgb

# Loading train, test, and sessions data
df_train = pd.read_csv('../input/train_users_2.csv')
df_test = pd.read_csv('../input/test_users.csv')
sessions = pd.read_csv('../input/sessions.csv')
labels = df_train['country_destination'].values
id_by_cntry = {c:df_train[df_train.country_destination == c]['id'] for c in np.unique(labels)}
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
train_test_cutoff = df_train.shape[0]

# Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

##### Aggregating sessions data (based on Abeer Jha post) #####
# only keeping sessions data with user ids found in train & test data
id_all = df_all['id']
sessions_rel = sessions[sessions.user_id.isin(id_all)]
'''
# Jaccard Similarity action_details
def jaccard(first, second):
  first = set(first)
  second = set(second)
  return len(first & second) / float(len(first | second))
  
booked_sets = {c:set(sessions_rel[sessions_rel.user_id.isin(id_by_cntry[c])].action_detail) for c in np.unique(labels)}
df_jss = pd.DataFrame(columns = [c+'_jss' for c in np.unique(labels)])
for c in np.unique(labels):
    def jss(act_det):
        return jaccard(act_det, booked_sets[c])
    df_jss[c+'_jss'] = sessions_rel.groupby(['user_id']).action_detail.apply(jss)
'''
# calculating total time elapsed
grp_by_sec_elapsed = sessions_rel.groupby(['user_id'])['secs_elapsed'].sum().reset_index()
grp_by_sec_elapsed.columns = ['user_id', 'secs_elapsed']

# aggregating by action and counting action_details
ct_action_detailXaction = pd.pivot_table(sessions_rel, index = ['user_id'],
                        columns = ['action'],
                        values = 'action_detail',
                        aggfunc=len, fill_value=0).reset_index()
ct_action_detailXaction.rename(
    columns = lambda x: x if (x == 'user_id') else x + "_action_detail_ct", 
    inplace = True
)
'''# aggregating by action_details and counting action_type
ct_action_typeXaction_detail = pd.pivot_table(sessions_rel, index = ['user_id'],
                        columns = ['action_detail'],
                        values = 'action_type',
                        aggfunc=len, fill_value=0).reset_index()
ct_action_typeXaction_detail.rename(
    columns = lambda x: x if (x == 'user_id') else x + "_actiontype_ct", 
    inplace = True
)'''                   
# aggregating by action_type and counting actions                        
ct_actionXaction_type = pd.pivot_table(sessions_rel, index = ['user_id'],
                             columns = ['action_type'],
                             values = 'action',
                             aggfunc=len, fill_value=0).reset_index()
ct_actionXaction_type.rename(
    columns = lambda x: x if (x == 'user_id') else x + "_action_ct", 
    inplace = True
)                          
# aggregating by device_type and counting actions                             
ct_actionXdevice_type = pd.pivot_table(sessions_rel, index = ['user_id'],
                             columns = ['device_type'],
                             values = 'action',
                             aggfunc=len, fill_value=0).reset_index()
ct_actionXdevice_type.rename(
    columns = lambda x: x if (x == 'user_id') else x + "_action_ct", #_device_action_ct
    inplace = True
)                                            
# aggregating total time elapsed by action_detail
sum_secsXaction_detail = pd.pivot_table(sessions_rel, index = ['user_id'],
                        columns = ['action_detail'],
                        values = 'secs_elapsed',
                        aggfunc=sum, fill_value=0).reset_index()
sum_secsXaction_detail.rename(
columns = lambda x: x if (x == 'user_id') else x + "_secs", 
inplace = True
)                           
# adding aggregated session features to dataframe                             
sessions_data = pd.merge(ct_actionXaction_type, ct_actionXdevice_type, 
                         on='user_id', how='inner')
sessions_data = pd.merge(sessions_data, ct_action_detailXaction, 
                         on='user_id',how='inner')
'''sessions_data = pd.merge(sessions_data, ct_actionXaction_detail, 
                         on='user_id',how='inner')'''
sessions_data = pd.merge(sessions_data, sum_secsXaction_detail, 
                         on='user_id',how='inner')                             
sessions_data = pd.merge(sessions_data, grp_by_sec_elapsed,
                         on='user_id', how='inner')
sessions_cols = sessions_data.columns.values
df_all = pd.merge(df_all, sessions_data, left_on='id', 
                  right_on='user_id', how='left')
'''df_all = pd.merge(df_all, df_jss.reset_index(), left_on='id', 
                  right_on='user_id', how='left')'''

# Removing id and date_first_booking
df_all = df_all.drop(['id', 'user_id', 'date_first_booking'], axis=1)

# Filling all nan with -1 
# tried imputing age with a rf, but did not improve results
df_all.age = df_all.age.fillna(max(df_all.age))
df_all = df_all.fillna(-1)

##### Feature engineering #####
# creating a 30d window before 5 major US holidays
holidays_tuples = holidays.US(years=[2010,2011,2012,2013,2014])
popular_holidays = ['Independence Day', 'Labor Day', 'Memorial Day']
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
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
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
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
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
'''df_all['age_sq'] = df_all['age']**2'''

# One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 
             'affiliate_channel', 'affiliate_provider', 'dac_year', 'dac_month', 
             'first_affiliate_tracked', 'signup_app', 'tfa_year', 'tfa_month',
             'first_device_type', 'first_browser', 'dac_day_of_wk', 'tfa_day_of_wk']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

'''
le = LabelEncoder()
y = le.fit_transform(labels) 

df_sess_trn = pd.concat([pd.DataFrame(y), df_all], axis=1).iloc[:train_test_cutoff,:]
df_sess_tst = df_all.iloc[train_test_cutoff:,:]
df_sess_trn = df_sess_trn[df_sess_trn.id.isin(sessions.user_id)]
df_sess_tst = df_sess_tst[df_sess_tst.id.isin(sessions.user_id)]
sess_ids = pd.concat([df_sess_trn['id'],df_sess_tst['id']]).reset_index(drop=True)
df_sess_trn = df_sess_trn.drop(['id', 'user_id', 'date_first_booking'], axis=1)
df_sess_tst = df_sess_tst.drop(['id', 'user_id', 'date_first_booking'], axis=1)
X_sess = df_sess_trn.iloc[:, 1:].values
y_sess = df_sess_trn.iloc[:train_test_cutoff, 0].values

X_tst_sess = df_sess_tst.iloc[:, 1:].values

params_sess = {'eta': 0.05, 
              'max_depth': 6,
              'subsample': 0.6, 
              'colsample_bytree': 0.7, 
              'objective': 'multi:softprob', 
              'num_class': 12,
              'eval_metric':'ndcg@5-',
              'seed':1234}

label2num2 = {label: i for i, label in enumerate(sorted(set(y_sess)))}
dtrain2 = xgb.DMatrix(X_sess, label=[label2num2[label] for label in y_sess])
clf_sess = xgb.train(params=params_sess, dtrain=dtrain2, num_boost_round=200)

y_pred_train = clf_sess.predict(xgb.DMatrix(X_sess), ntree_limit=clf_sess.best_iteration)
y_pred_test = clf_sess.predict(xgb.DMatrix(X_tst_sess), ntree_limit=clf_sess.best_iteration)
y_pred_sess = np.vstack((y_pred_train, y_pred_test))

meta_feat = pd.concat([sess_ids,pd.DataFrame(y_pred_sess)], axis =1)
        
df_meta = pd.merge(df_all, meta_feat, on = 'id', how = 'left')
drop_cols = np.hstack((['id', 'user_id', 'date_first_booking'], sessions_cols))
df_meta.drop(drop_cols, axis = 1, inplace= True)
df_meta = df_meta.fillna(-1)

X = df_meta.values[:train_test_cutoff]
X_tst_meta = df_meta.values[train_test_cutoff:]

params_stck = {'eta': 0.05, 
              'max_depth': 5,
              'subsample': 0.7, 
              'colsample_bytree': 1, 
              'objective': 'multi:softprob', 
              'num_class': 12,
              'eval_metric':'ndcg@5-',
              'seed':1234}

label2num1 = {label: i for i, label in enumerate(sorted(set(y)))}
dtrain1 = xgb.DMatrix(X, label=[label2num1[label] for label in y]) 
clf = xgb.train(params=params_stck, dtrain=dtrain1, num_boost_round=200)

y_pred = clf.predict(xgb.DMatrix(X_tst_meta), ntree_limit=clf.best_iteration)
'''

'''
X_no_sess = df_all.drop(sessions_cols, axis = 1)[:train_test_cutoff]
X_no_sess = X_no_sess.drop(['id', 'date_first_booking'], axis=1).values
le = LabelEncoder()
y = le.fit_transform(labels) 

df_sess = pd.concat([pd.DataFrame(y), df_all.iloc[:train_test_cutoff,:]], axis=1)
df_sess = df_sess[df_sess.id.isin(sessions.user_id)]
df_sess = df_sess.drop(['id', 'user_id', 'date_first_booking'], axis=1)
X_sess = df_sess.iloc[:, 1:].values
y_sess = df_sess.iloc[:, 0].values

X_test_no_sess = df_all.drop(np.hstack((['id', 'user_id', 'date_first_booking'],sessions_cols)), axis=1).values[train_test_cutoff:]
X_test = df_all.drop(['id', 'user_id', 'date_first_booking'], axis=1).values[train_test_cutoff:]

params1 = {'eta': 0.05, 
              'max_depth': 5,
              'subsample': 0.7, 
              'colsample_bytree': 1, 
              'objective': 'multi:softprob', 
              'num_class': 12,
              'eval_metric':'ndcg@5-',
              'seed':1234}

label2num1 = {label: i for i, label in enumerate(sorted(set(y)))}
dtrain1 = xgb.DMatrix(X_no_sess, label=[label2num1[label] for label in y]) 
clf = xgb.train(params=params1, dtrain=dtrain1, num_boost_round=200)

y_pred1_train = clf.predict(xgb.DMatrix(X_sess), ntree_limit=clf.best_iteration)
y_pred1_test = clf.predict(xgb.DMatrix(X_test_no_sess), ntree_limit=clf.best_iteration)

params2 = {'eta': 0.05, 
              'max_depth': 6,
              'subsample': 0.6, 
              'colsample_bytree': 0.7, 
              'objective': 'multi:softprob', 
              'num_class': 12,
              'eval_metric':'ndcg@5-',
              'seed':1234}

label2num2 = {label: i for i, label in enumerate(sorted(set(y_sess)))}
dtrain2 = xgb.DMatrix(X_sess, label=[label2num2[label] for label in y_sess])
clf_sess = xgb.train(params=params2, dtrain=dtrain2, num_boost_round=200)

y_pred2_train = clf_sess.predict(xgb.DMatrix(X_sess), ntree_limit=clf_sess.best_iteration)
y_pred2_test = clf_sess.predict(xgb.DMatrix(X_test), ntree_limit=clf_sess.best_iteration)

X_meta_train = np.hstack((y_pred1_train, y_pred2_train))
X_meta_test = np.hstack((y_pred1_test, y_pred2_test))

params3 = {'eta': 0.1, 
              'max_depth': 5,
              'subsample': 1.0, 
              'colsample_bytree': 1.0, 
              'objective': 'multi:softprob', 
              'num_class': 12,
              'eval_metric':'ndcg@5-',
              'seed':1234}

dtrain3 = xgb.DMatrix(X_meta_train, label=[label2num1[label] for label in y_sess]) 
ens_clf = xgb.train(params=params3, dtrain=dtrain3, num_boost_round=50)

y_pred = ens_clf.predict(xgb.DMatrix(X_meta_test), ntree_limit=ens_clf.best_iteration).reshape(df_test.shape[0],12) 
'''

# performing feature selection based on xgb.get_fscore
feat_keep = pd.read_csv('features-kwu2u.csv')
df_all = df_all[feat_keep.feature.values]

# Splitting train and test
X = df_all.iloc[:train_test_cutoff,:]
le = LabelEncoder()
y = le.fit_transform(labels)   
X_test = df_all.iloc[train_test_cutoff:,:]


# Classifier
opt_params = {'eta': 0.05, 
              'max_depth': 6,
              'subsample': 0.7, 
              'colsample_bytree': 0.7, 
              'objective': 'multi:softprob',
              'num_class': 12,
              'eval_metric':'ndcg@5',
              'seed':1234}
              
weight = [1.2 if (row.tfa_month_7 == 1 or row.tfa_month_8 == 1 or row.tfa_month_9 == 1) else 1 for ix, row in X.iterrows()]
label2num = {label: i for i, label in enumerate(sorted(set(y)))}
dtrain = xgb.DMatrix(X, label=[label2num[label] for label in y], weight=weight)
bst = xgb.train(params=opt_params, dtrain=dtrain, num_boost_round=200)

y_pred = bst.predict(xgb.DMatrix(X_test), 
                     ntree_limit=bst.best_iteration
                ).reshape(df_test.shape[0],12) 
             
# Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

# Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)