from Munger import clean_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer

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

def dcg_score(y_true, y_score, k=5):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_score(ground_truth, predictions, k=5):
    lb = LabelBinarizer()
    T = lb.fit_transform(ground_truth)

    scores = []

    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        score = float(actual) / float(best)
        scores.append(score)

    return np.mean(scores)

# NDCG Scorer function
ndcg_scorer = make_scorer(ndcg_score, needs_proba=True)

rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X)
clf = SGDClassifier(loss='log', n_jobs=-1, n_iter = 50)
scores = cross_val_score(clf, X_features, y, cv=3, scoring=ndcg_scorer)

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