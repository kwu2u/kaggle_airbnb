# This script is based on a forum post by Sandro
# and has been put into OOP form based on a forum post by Simple Liquids
# note: tree ensembles are invariant to scaling so no normalization necessary,
# one hot encoding categorical features

import numpy as np
import pandas as pd
import datetime
import holidays
from sklearn.preprocessing import LabelEncoder
import copy

class Munger():
    def __init__(self):
        np.random.seed(0)
        self.le = LabelEncoder()
    
        #Loading data
        df_train = pd.read_csv('../input/train_users_2.csv')
        df_test = pd.read_csv('../input/test_users.csv')
        sessions = pd.read_csv('../input/sessions.csv')
        self.labels = df_train['country_destination'].values
        df_train = df_train.drop(['country_destination'], axis=1)
        self.id_test = df_test['id']
        self.train_test_cutoff = df_train.shape[0]
        
        # Creating a DataFrame with train+test data
        self.df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
        
        ##### Aggregating sessions data (based on Abeer Jha post) #####
        # only keeping sessions data with user ids found in train & test data
        id_all = self.df_all['id']
        sessions_rel = sessions[sessions.user_id.isin(id_all)]
        
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
        ''' this is mostly captured by summing secs_elapsed by action_detail                   
        # aggregating by action_details and counting actions
        ct_actionXaction_detail = pd.pivot_table(sessions_rel, index = ['user_id'],
                                columns = ['action_detail'],
                                values = 'action',
                                aggfunc=len, fill_value=0).reset_index()
        ct_actionXaction_detail.rename(
            columns = lambda x: x if (x == 'user_id') else x + "_action_ct", 
            inplace = True
        )
        '''                  
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
            columns = lambda x: x if (x == 'user_id') else x + "_action_ct", 
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
        '''
        sessions_data = pd.merge(sessions_data, ct_actionXaction_detail, 
                                 on='user_id',how='inner')
        '''
        sessions_data = pd.merge(sessions_data, sum_secsXaction_detail, 
                                 on='user_id',how='inner')                             
        sessions_data = pd.merge(sessions_data, grp_by_sec_elapsed,
                                 on='user_id', how='inner')
        self.df_all = pd.merge(self.df_all, sessions_data, left_on='id', 
                          right_on='user_id', how='left')
        
        # Removing id and date_first_booking
        self.df_all = self.df_all.drop(['id', 'user_id', 'date_first_booking'], axis=1)
        # Filling all nan with -1 
        # tried imputing age with a rf, but did not improve results
        self.df_all = self.df_all.fillna(-1)
        
    def engineer_features(self):
        # creating a 30d window before 5 major US holidays
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
        dac = pd.to_datetime(self.df_all['date_account_created'])
        self.df_all['dac_year'] = dac.apply( lambda x: x.year)
        self.df_all['dac_month'] = dac.apply( lambda x: x.month)
        self.df_all['dac_day'] = dac.apply( lambda x: x.day)
        self.df_all['dac_holiday_30'] = dac.isin(holiday_30)
        
        dac_day_of_wk = []
        for date in dac:
            dac_day_of_wk.append(date.weekday())
        self.df_all['dac_day_of_wk'] = pd.Series(dac_day_of_wk)
        
        self.df_all = self.df_all.drop(['date_account_created'], axis=1)
        
        # timestamp_first_active
        def convert_time(time_string):
            return datetime.datetime.strptime(str(time_string), '%Y%m%d%H%M%S')
            
        tfa = self.df_all.timestamp_first_active.apply(convert_time)  
        self.df_all['tfa_year'] = tfa.apply( lambda x: x.year)
        self.df_all['tfa_month'] = tfa.apply( lambda x: x.month)
        self.df_all['tfa_day'] = tfa.apply( lambda x: x.day)
        #self.df_all['tfa_hour'] = tfa.apply( lambda x: x.hour)
        self.df_all['tfa_holiday_30'] = tfa.isin(holiday_30)
        
        tfa_day_of_wk = []
        for date in tfa:
            tfa_day_of_wk.append(date.weekday())
        self.df_all['tfa_day_of_wk'] = pd.Series(tfa_day_of_wk)
        
        self.df_all = self.df_all.drop(['timestamp_first_active'], axis=1)
        
        # Age 
        # valid range (14-100), calculate birth date (1919-1995), else -1
        av = self.df_all.age.values
        self.df_all['age'] = np.where(np.logical_and(av>1919, av<1995), 2015-av, av)
        self.df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)
        
    def one_hot_encode(self):
        ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 
                     'affiliate_channel', 'affiliate_provider', 'dac_year', 'dac_month', 
                     'first_affiliate_tracked', 'signup_app', 'tfa_year', 'tfa_month',
                     'first_device_type', 'first_browser', 'dac_day_of_wk', 'tfa_day_of_wk']
        for f in ohe_feats:
            df_all_dummy = pd.get_dummies(self.df_all[f], prefix=f)
            self.df_all = self.df_all.drop([f], axis=1)
            self.df_all = pd.concat((self.df_all, df_all_dummy), axis=1)
            
    def label_transformer(self):
        return self.le.fit_transform(self.labels)
        
    def label_inverse_transformer(self, labels):
        return self.le.inverse_transform(labels)
        
    def data_split(self):
        vals = self.df_all.values
        X = vals[:self.train_test_cutoff]
        X_test = vals[self.train_test_cutoff:]
        return X, X_test
        
    def get_train_data(self):
        df_all = copy.deepcopy(self.df_all.iloc[:self.train_test_cutoff:])
        df_all['y'] = self.labels
        return df_all
        
    def select_features(self, feature_dataframe):
        # feature_dataframe has 2 columns: feature & fscore
        # generate by feature_selection.py
        self.df_all = self.df_all[feature_dataframe.feature.values]

def clean_data():
    M = Munger()
    M.engineer_features()
    M.one_hot_encode()
    return M