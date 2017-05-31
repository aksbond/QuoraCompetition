import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import log_loss
from numpy import inf

with open('./Data/Feature sets/Basic_features_w_W2vec.pkl', 'rb') as input:
    full_data = pickle.load(input)

full_data.shape
full_data.describe()

from numpy import inf
full_data.loc[full_data["wmd"] == inf, "wmd"] = 10
full_data.loc[full_data["norm_wmd"] == inf, "norm_wmd"] = 10


#################################################################
# Oversampling the data here
#################################################################

pos_train = full_data.loc[((full_data["train_ind"] == 1) & (full_data["is_duplicate"] == 1)),:]
neg_train = full_data.loc[((full_data["train_ind"] == 1) & (full_data["is_duplicate"] == 0)),:]

# Now we oversample the negative class
# There is likely a much more elegant way to do this...
p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print(len(pos_train) / (len(pos_train) + len(neg_train)))

x_train = pd.concat([pos_train, neg_train], axis = 0)
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train


# Finally, we split some of the data off for validation
X_train_sample, X_test_sample, Y_train_sample, Y_test_sample = train_test_split(x_train, y_train, test_size=0.3, random_state=321)

drop_columns = ['id', 'is_duplicate', 'question1', 'question2', 'q1_n_q2', 'train_ind']
X_train_sample = X_train_sample.drop(drop_columns,axis=1).values
X_test_sample = X_test_sample.drop(drop_columns,axis=1).values

X_train = x_train.drop(drop_columns,axis = 1).values
Y_train = y_train

X_test = full_data.loc[full_data.train_ind == 0, :]
X_test = X_test.drop(drop_columns,axis=1).values
"""
# ------------- TRAIN MODEL ON 50% DATA --------------------------
#np.random.seed(321)
#train_sample, test_sample = train_test_split(full_data.loc[full_data.train_ind == 1,:], stratify = full_data.loc[full_data.train_ind == 1,"train_ind"], train_size = 0.7)
#train_sample.describe()


Y_train_sample = train_sample.loc[:,"is_duplicate"].values
Y_test_sample = test_sample.loc[:,"is_duplicate"].values




full_data.dtypes[full_data.dtypes == "float64"]
full_data.columns
drop_columns = ['id', 'is_duplicate', 'question1', 'question2', 'q1_n_q2', 'train_ind']
X_train_sample = train_sample.drop(drop_columns,axis=1).values
X_test_sample = test_sample.drop(drop_columns,axis=1).values
#X_train = train_data.drop(drop_columns,axis=1).values
#X_test = full_data.loc[full_data.train_ind == 0,:].drop(drop_columns,axis=1).values
X_test.shape
X_train.shape
X_train_sample.shape
X_test_sample.shape
#######################################################################
################

##------------------------------------------------------------------------------------------------
"""


xgb_clf = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=1000, objective='binary:logistic', subsample=0.5, colsample_bytree=0.5, seed=0)
eval_set  = [(X_train_sample, Y_train_sample), (X_test_sample, Y_test_sample)]
xgb_clf.fit(X_train_sample, Y_train_sample, eval_set = eval_set, eval_metric = "logloss", early_stopping_rounds= 50)
Y_pred_sample = xgb_clf.predict_proba(X_test_sample)
log_loss(Y_test_sample,Y_pred_sample)

### 551 iterations
del(xgb_clf)
del(eval_set)
del(full_data)

xgb_clf = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=1500, objective='binary:logistic', subsample=0.5, colsample_bytree=0.5, seed=0)
#eval_set  = [(X_train_sample, Y_train_sample), (X_test_sample, Y_test_sample)]
xgb_clf.fit(X_train, Y_train, eval_metric = "logloss")
Y_pred = xgb_clf.predict_proba(X_test)

len([x for x in Y_pred[:,0] if x > 0.5])
len([x for x in Y_pred[:,0] if x <= 0.5])

Y_pred = pd.DataFrame(Y_pred)
Y_pred = Y_pred.reset_index(drop=False)
Y_pred.head(5)
Y_pred = Y_pred.drop(0,axis=1)
Y_pred.columns = ["test_id","is_duplicate"]

Y_pred.to_csv("./Data/Submissions/Basic_features.csv", index = False)

del(Y_pred)



import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Build a classification task using 3 informative features
# Build a forest and compute the feature importances
rf.fit(X_train_sample, Y_train_sample_clarity)
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

all_features = list({k for k in full_data.columns if k not in drop_columns})

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train_sample.shape[1]):
    print("%d. %s (%f)" % (f + 1, all_features[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train_sample.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train_sample.shape[1]), indices)
plt.xlim([-1, X_train_sample.shape[1]])
plt.show()


train_data.loc[:,["unq_mes_str_title", "concise"]].describe()
pd.crosstab(train_data.clarity, train_data.unq_color_str_title)

