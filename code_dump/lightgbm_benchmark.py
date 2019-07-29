"""
meta_data = [{'dataset': 'adult', 'nclass': 2, 'label': 'income'},
 {'dataset': 'covtype', 'nclass': 7, 'label': 'X55'},
 {'dataset': 'dna', 'nclass': 3, 'label': 'Class'},
 {'dataset': 'glass', 'nclass': 6, 'label': 'Class'},
 {'dataset': 'letter', 'nclass': 26, 'label': 'lettr'},
 {'dataset': 'sat', 'nclass': 8, 'label': 'X37'},
 {'dataset': 'shuttle', 'nclass': 7, 'label': 'Class'},
 {'dataset': 'simplemandelon', 'nclass': 2, 'label': 'y'},
 {'dataset': 'soybean', 'nclass': 19, 'label': 'Class'},
 {'dataset': 'yeast', 'nclass': 10, 'label': 'yeast'}
]

for mdat in meta_data:
    obj = "binary" if mdat['nclass'] <= 2 else "multiclass"
    dat = mdat['dataset']
    lab = mdat['label']
    print(f"./run_gbm.sh lightgbm_benchmark.py {obj} {dat} gbdt 5 100 {lab}")
    print(f"./run_gbm.sh lightgbm_benchmark.py {obj} {dat} gbdt 5 5 {lab}")
    print(f"./run_gbm.sh lightgbm_benchmark.py {obj} {dat} gbdt 5 15 {lab}")
    print(f"./run_gbm.sh lightgbm_benchmark.py {obj} {dat} rf 5 100 {lab}")
    print(f"./run_gbm.sh lightgbm_benchmark.py {obj} {dat} rf 5 5 {lab}")
    print(f"./run_gbm.sh lightgbm_benchmark.py {obj} {dat} rf 5 15 {lab}")


"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import sys
import os

from sklearn.metrics import roc_auc_score
import sklearn.metrics
import time

def split_pred(df, label):
    return df[[x for x in df.columns if x != label]], df[label]

# add sys.args

config_dict = {
    'rf': {
        'boosting_type': 'rf',
        'bagging_freq': 1,
        'bagging_fraction': 0.7,
        'feature_fraction': 0.7
    },
    'gbdt': {
        'boosting_type': 'gbdt'
    }
}

if len(sys.argv) == 1:
    objective="binary"
    dataset="adult"
    boost_type = 'rf'
    depth = 5
    n_estimators = 100
    label = "income"
else:
    # ./run_gbm.sh lightgbm_benchmark.py binary adult gbdt 5 100 income
    print(sys.argv)
    _, objective, dataset, boost_type, depth, n_estimators, label = sys.argv
    depth = int(depth)
    n_estimators = int(n_estimators)

save_dir = "lightgbm_benchmark"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

train_adult = pd.read_csv(f"clean_data/{dataset}_train_scale.csv")
test_adult = pd.read_csv(f"clean_data/{dataset}_test_scale.csv")
x_train, y_train = split_pred(train_adult, label)
x_test, y_test = split_pred(test_adult, label)

class_config = config_dict[boost_type]
class_config['objective'] = objective
class_config['num_leaves'] = 2**depth
class_config['n_estimators'] = n_estimators

gbm = lgb.LGBMClassifier(**class_config)

start = time.time()
gbm.fit(x_train, y_train,
        eval_set=[(x_test, y_test)],
        eval_metric='logloss',
        early_stopping_rounds=None)
end = time.time()

y_pred_train = gbm.predict_proba(x_train)
y_pred = gbm.predict_proba(x_test)
print("Performance (logloss) is: {}".format(sklearn.metrics.log_loss(y_test, y_pred)))
print("Performance (accuracy) is: {}".format(sklearn.metrics.accuracy_score(y_test, gbm.predict(x_test))))
print("Time is: {}".format(end-start))

df = pd.DataFrame({
    'dataset': [dataset, dataset],
    'logloss': [sklearn.metrics.log_loss(y_train, y_pred_train), sklearn.metrics.log_loss(y_test, y_pred)],
    'accuracy': [sklearn.metrics.accuracy_score(y_train, gbm.predict(x_train)), sklearn.metrics.accuracy_score(y_test, gbm.predict(x_test))],
    'boosttype': [boost_type, boost_type],
    'depth': [depth, depth],
    'n_estimators': [n_estimators, n_estimators],
    'train_test': ["train", "test"]
})


df.to_csv(f"{save_dir}/{dataset}_{boost_type}{n_estimators}.csv", index=False)
