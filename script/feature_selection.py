from Munger import clean_data
import pandas as pd
import xgboost as xgb

M = clean_data()

feat_keep = pd.read_csv('features.csv')
M.select_features(feat_keep)

X, X_test = M.data_split()
y = M.label_transformer()

opt_params = {'eta': 0.2,
              'max_depth': 6,
              'subsample': 0.5,
              'colsample_bytree': 0.7,
              'objective': 'multi:softprob',
              'num_class': 12,
              'eval_metric':'ndcg',
              'seed':1234}
              
label2num = {label: i for i, label in enumerate(sorted(set(y)))}
dtrain = xgb.DMatrix(X, label=[label2num[label] for label in y])
bst = xgb.train(params=opt_params, dtrain=dtrain, num_boost_round=30)
#xgb.plot_importance(bst)

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()

create_feature_map(list(M.df_all.columns.values))
importance = bst.get_fscore(fmap='xgb.fmap')
importance_df = pd.DataFrame(importance.items(), columns=['feature','fscore'])
importance_df.to_csv('features.csv',index=False)

'''
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
'''