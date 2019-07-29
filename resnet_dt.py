"""
Sample implementation of ResNet-DT

To run:

```
python resnet_dt.py
```

This file runs a variant of the Mandelon dataset as per scikit-learn
"""

import os
import sys

import pandas as pd

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

import tensorflow as tf

from keras.layers import Dense, Flatten, Input, Lambda
from keras.models import Model

from keras import backend as K
from keras import initializers, layers
import keras

from keras.utils import to_categorical
from keras.constraints import min_max_norm, non_neg, max_norm
from keras.callbacks import Callback
import time
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

import tensorflow as tf

from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

from keras import backend as K
from keras import initializers, layers

from keras.utils import to_categorical
from keras.constraints import non_neg, max_norm

from keras.initializers import Zeros
from keras.constraints import Constraint
from keras.regularizers import l2

from gbm_util import * 

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

def split_pred(df, label):
    return df[[x for x in df.columns if x != label]], df[label]

# add sys.args
if len(sys.argv) == 1:
    ntree=5
    last_only=True
else:
    # run with <script> 5, 1
    # run with <script> 15, 1
    # run with <script> 5, 0
    # run with <script> 15, 0
    _, ntree, last_only = sys.argv
    last_only = last_only == "1"
    ntree = int(ntree)


depth = 5
dim_size = 20
num_class=2

path = "clean_data"
from sklearn.datasets import make_classification

X, y = make_classification()
y = to_categorical(y)

save_dir = os.path.join(os.getcwd(), 'saved_models')
save_dir = "mandelon_benchmark"
model_name = 'mandelon_model_gbm.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

tree = Tree() # this keeps the state of the current decision tree...
input_dim = dim_size


nepochs = 200

# Prepare callbacks for model saving and for learning rate adjustment.
time_cb = TimingCallback()

print("Attempting to load model...")
#model = gbdt_model(97, n_tree=5, depth=3)
# dim_size, n_tree=3, depth=3, sample=0.7, num_class=10, last_only=False
model, n_layer = gbm_model2(dim_size, n_tree=ntree, depth=depth, sample=1.0, num_class=num_class, last_only=last_only)
#model, n_layer = rf_model2(97, n_tree=5, depth=5, sample=1.0, num_class=2, last_only=True)
model.compile(optimizer=Adam(lr=lr_schedule(0), clipnorm=1.), 
            loss='categorical_crossentropy',
            metrics=['accuracy'])

print("Model has {} layers".format(len(model.layers)))
hist = model.fit(X, [y for _ in range(n_layer)], epochs=nepochs, verbose=2, 
                 validation_data=(X, [y for _ in range(n_layer)]), # in actual run it was different data
                 callbacks=[time_cb])

hist_df = pd.DataFrame(hist.history)
hist_df['times'] = time_cb.times[-hist_df.shape[0]:]
print(hist_df.iloc[-1])

# this normally hits 100% accuracy