# contest: HackerRank Machine Learning CodeSprint
# challenge: Predict Email Opens
# date: September 4th, 2016
# username: caiotaniguchi
# name: Caio Taniguchi
# email: caiotaniguchi@gmail.com

# Importing packages
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score


print('Loading data files...')
train = pd.read_csv('training_dataset.csv')
test = pd.read_csv('test_dataset.csv')

print('Transforming features...')
# Features to drop or transform
to_drop_train = ['click_time', 'clicked', 'open_time', 'unsubscribe_time', 'unsubscribed']
to_drop_both = ['contest_login_count_1_days', 'contest_participation_count_1_days', 'ipn_read_1_days', 
                'ipn_count_1_days', 'mail_type', 'submissions_count_contest_1_days', 'submissions_count_master_1_days',
                'submissions_count_1_days', 'ipn_read_7_days', 'contest_login_count_7_days', 
                'contest_participation_count_7_days', 'forum_expert_count', 'hacker_confirmation']
to_transform = ['sent_time', 'last_online', 'hacker_created_at']

train['mail_category'] = pd.to_numeric(train['mail_category'].str.split('_').str.get(2))
test['mail_category'] = pd.to_numeric(test['mail_category'].str.split('_').str.get(2))
train['mail_category'].fillna(1, inplace=True)
test['mail_category'].fillna(1, inplace=True)

train['sent_time_last_online_diff'] = train['sent_time'] - train['last_online']
test['sent_time_last_online_diff'] = test['sent_time'] - test['last_online']
train['sent_time_last_online_diff'].fillna(train['sent_time_last_online_diff'].median(), inplace=True)
test['sent_time_last_online_diff'].fillna(train['sent_time_last_online_diff'].median(), inplace=True)
train['sent_time_created_at_diff'] = train['sent_time'] - train['hacker_created_at']
test['sent_time_created_at_diff'] = test['sent_time'] - test['hacker_created_at']

hacker_tz_mode = train['hacker_timezone'].mode().values[0]
train['hacker_timezone'].fillna(hacker_tz_mode, inplace=True)
train['last_online'].fillna(train['last_online'].median(), inplace=True)
test['hacker_timezone'].fillna(hacker_tz_mode, inplace=True)
test['last_online'].fillna(train['last_online'].median(), inplace=True)

train['tz_sent_time'] = train['sent_time'] + train['hacker_timezone']
test['tz_sent_time'] = test['sent_time'] + test['hacker_timezone']
train['tz_last_online'] = train['last_online'] + train['hacker_timezone']
test['tz_last_online'] = test['last_online'] + test['hacker_timezone']
train['tz_hacker_created_at'] = train['hacker_created_at'] + train['hacker_timezone']
test['tz_hacker_created_at'] = test['hacker_created_at'] + test['hacker_timezone']

sent_time = pd.DatetimeIndex(pd.to_datetime(train['sent_time'], unit='s'))
last_online = pd.DatetimeIndex(pd.to_datetime(train['last_online'], unit='s'))
hacker_created_at = pd.DatetimeIndex(pd.to_datetime(train['last_online'], unit='s'))
train['sent_time_day'] = sent_time.dayofyear
train['sent_time_wkday'] = sent_time.weekday
train['sent_time_hour'] = sent_time.hour
train['last_online_day'] = last_online.dayofyear
train['last_online_wkday'] = last_online.weekday
train['last_online_hour'] = last_online.hour
train['hacker_created_at_wkday'] = hacker_created_at.weekday
train['hacker_created_at_hour'] = hacker_created_at.hour

sent_time = pd.DatetimeIndex(pd.to_datetime(test['sent_time'], unit='s'))
last_online = pd.DatetimeIndex(pd.to_datetime(test['last_online'], unit='s'))
hacker_created_at = pd.DatetimeIndex(pd.to_datetime(test['last_online'], unit='s'))
test['sent_time_day'] = sent_time.dayofyear
test['sent_time_wkday'] = sent_time.weekday
test['sent_time_hour'] = sent_time.hour
test['last_online_day'] = last_online.dayofyear
test['last_online_wkday'] = last_online.weekday
test['last_online_hour'] = last_online.hour
test['hacker_created_at_wkday'] = hacker_created_at.weekday
test['hacker_created_at_hour'] = hacker_created_at.hour


train.drop(to_drop_both, axis=1, inplace=True)
test.drop(to_drop_both, axis=1, inplace=True)


print('Preparing the data for training...')
spl = StratifiedShuffleSplit(train['opened'], test_size=0.1, n_iter=1, random_state=42)
X_train = None
X_val = None

for train_index, test_index in spl:
    X_train = train.iloc[train_index, :]
    X_val = train.iloc[test_index, :]
    
    X_mean_opened_user = X_train.groupby(['user_id'], as_index=False)['opened'].agg({
        'mean_opened_user' : np.mean
    })
    X_mean_opened_mail_id = X_train.groupby(['mail_id'], as_index=False)['opened'].agg({
        'mean_opened_mail_id' : np.mean
    })

    X_train = X_train.merge(X_mean_opened_user, on=['user_id'], how='left', sort=False)
    X_train = X_train.merge(X_mean_opened_mail_id, on=['mail_id'], how='left', sort=False)
    X_val = X_val.merge(X_mean_opened_user, on=['user_id'], how='left', sort=False)
    X_val = X_val.merge(X_mean_opened_mail_id, on=['mail_id'], how='left', sort=False)
        
    X_sum_unsubscribed_mail = X_train.groupby('mail_id', as_index=False)['unsubscribed'].sum()
    X_sum_unsubscribed_mail.rename(columns={'unsubscribed': 'unsubscribed_mail'}, inplace=True)
    X_train = X_train.merge(X_sum_unsubscribed_mail, on='mail_id', how='left')
    X_val = X_val.merge(X_sum_unsubscribed_mail, on=['mail_id'], how='left', sort=False)

dtrain = xgb.DMatrix(data=X_train.drop(['mail_id', 'user_id', 'opened'] + to_drop_train, axis=1), 
                     label=X_train['opened'])
dval = xgb.DMatrix(data=X_val.drop(['mail_id', 'user_id', 'opened'] + to_drop_train, axis=1), label=X_val['opened'])

eta = 0.05
max_depth = 10
subsample = 0.8
colsample_bytree = 0.7
min_child_weight = 1
gamma = 0
lmbd = 1
alpha = 0
watchlist = [(dtrain, 'train'), (dval, 'val')]

print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'
      .format(eta, max_depth, subsample, colsample_bytree))

params = {
    "objective": "reg:logistic",
    "booster" : "gbtree",
    "eval_metric": "auc",
    "eta": eta,
    "max_depth": max_depth,
    "subsample": subsample,
    "colsample_bytree": colsample_bytree,
    "min_child_weight": min_child_weight,
    "gamma": gamma,
    "lambda": lmbd,
    "alpha": alpha,
    "silent": 1,
    "seed": 42,
}
num_boost_round = 20

clf = xgb.train(params, dtrain, num_boost_round, watchlist, verbose_eval=True)

print('Optimizing threshold...')
predictions_train = clf.predict(dtrain)
predictions_val = clf.predict(dval)
scores = []

for i in np.arange(0.2, 0.5, 0.001):
    scores.append({
            'threshold': i,
            'train_score': f1_score(X_train['opened'], predictions_train > i),
            'val_score': f1_score(X_val['opened'], predictions_val > i)
        })

print('Generating predictions...')
def mergeLeftInOrder(x, y, on=None):
    x = x.copy()
    x['order'] = np.arange(len(x))
    return x.merge(y, how='left', on=on).set_index('order').ix[np.arange(len(x)), :]

X_test = mergeLeftInOrder(test, X_mean_opened_user, on='user_id')
X_test = mergeLeftInOrder(X_test, X_mean_opened_mail_id, on='mail_id')
X_test = mergeLeftInOrder(X_test, X_sum_unsubscribed_mail, on='mail_id')
X_test['mean_opened_user'].fillna(X_mean_opened_user.median(), inplace=True)
X_test['mean_opened_mail_id'].fillna(X_mean_opened_mail_id.median(), inplace=True)
X_test['unsubscribed_mail'].fillna(0, inplace=True)

scores = pd.DataFrame(scores)
tht = scores.loc[scores['train_score'] == scores['train_score'].max(), 'threshold']
thv = scores.loc[scores['val_score'] == scores['val_score'].max(), 'threshold']
th = ((tht.values[0] + thv.values[0])/2) - 0.01
predictions = clf.predict(xgb.DMatrix(data=X_test.drop(['mail_id', 'user_id',], axis=1)))
submission = pd.DataFrame((predictions > th).astype(int))

print('Writing to prediction.csv file...')
submission.to_csv('prediction.csv', header=False, index=False)