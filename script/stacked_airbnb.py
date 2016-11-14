from Munger import clean_data
import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer

M = clean_data()

feat_keep = pd.read_csv('features.csv')
M.select_features(feat_keep)

X, X_test = M.data_split()
y = M.label_transformer()

############################ NDCG Scorer ############################
def dcg_score(y_true, y_score, k=5):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_score(ground_truth, predictions, k=5):
    lb = LabelBinarizer()
    lb.fit(range(len(predictions) + 1))
    T = lb.transform(ground_truth)

    scores = []

    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        score = float(actual) / float(best)
        scores.append(score)

    return np.mean(scores)

# NDCG Scorer function
ndcg_scorer = make_scorer(ndcg_score, needs_proba=True, k=5)                          

############################ XGBoost ############################
from xgboost.sklearn import XGBClassifier
xgbst = XGBClassifier(
    objective = 'multi:softprob',
    n_estimators = 200,
    learning_rate = 0.05,
    max_depth = 6,
    subsample = 0.7,
    colsample_bytree = 0.7,
    seed = 0
    )

xgbst.fit(X,y)
xgb_y_pred = xgbst.predict_proba(X_test)

############################ Random Forest ############################
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,random_state=42)
rf = RandomForestClassifier(n_estimators = 500, n_jobs = -1, 
                            random_state = 1234, class_weight='balanced_subsample')
rf.fit(X_train, y_train)
sig_rf = CalibratedClassifierCV(rf, method='sigmoid', cv='prefit')                           
# scores = cross_val_score(sig_rf, X, y, cv=3, scoring=ndcg_scorer)
sig_rf.fit(X_val,y_val)
rf_y_pred = sig_rf.predict_proba(X_test) 

############################ XGBoost Ensemble ############################
xgb_ensemble = XGBClassifier(
    objective = 'multi:softprob',
    n_estimators = 50,
    learning_rate = 0.2,
    max_depth = 4,
    subsample = 0.7,
    colsample_bytree = 1.0,
    seed = 0
    )

xgb_y_pred_train = xgbst.predict_proba(X)
rf_y_pred_train = sig_rf.predict_proba(X)
metafeatures = np.hstack((xgb_y_pred_train,rf_y_pred_train))
xgb_ensemble.fit(metafeatures,y)
y_pred = xgb_ensemble.predict_proba(np.hstack((xgb_y_pred,rf_y_pred))).reshape(X_test.shape[0],12) 

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