import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from keras import backend as K
from keras import initializers, layers
from keras.utils import to_categorical
from keras.constraints import non_neg, max_norm
from keras.initializers import Zeros
from keras.constraints import Constraint

import tensorflow as tf

from decision_tree import *

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

dim_size = 97
num_class=2
print(f"selected params - dim: {dim_size}, nclass: {num_class}, ntree: {ntree} depth: {depth}, last: {last_only}")

path = "clean_data"
train_adult = pd.read_csv(path+'/adult_train_scale.csv')
test_adult = pd.read_csv(path+'/adult_test_scale.csv')

x, y = split_pred(train_adult, "income")
x_test, y_test = split_pred(test_adult, "income")
y = to_categorical(y)
y_test = to_categorical(y_test)

save_dir = os.path.join(os.getcwd(), 'saved_models')
save_dir = "adult_benchmark"

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

tree = Tree() # this keeps the state of the current decision tree...
input_dim = dim_size

nepochs = 200

class TimingCallback(Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()
    
    def on_epoch_end(self, batch, logs={}):
        # write stuff to disc here...
        self.times.append(time.time() - self.epoch_time_start)


def gen_states(tree, tree_list=[0], target_idx=None, return_tree_list=False):
    def size_0(dct):
        for key, val in dct.items():
            if len(val) > 0:
                return False
        return True
    
    tree_index = max(tree_list)
    if target_idx is None:
        curr_list = [tree_index+1, tree_index+2, tree_index+3]
    else:
        curr_list = [tree_index+1, target_idx, tree_index+2]
    
    tree_list.extend(curr_list)
    d0, s0 = tree.prune()
    d1 = tree.tree.copy()
    d2, s2 = tree.graft()
    
    if size_0(d0):
        # reset
        d0 = Tree().tree.copy()    
    
    state_info = {'prune': (d0, curr_list[0]),
            'base': (d1, curr_list[1]),
            'graft': (d2, curr_list[2]),
            'state': {
                'prune': s0, 'graft': s2
            }}
    if return_tree_list:
        return state_info, tree_list, curr_list
    else:
        return state_info


# In[6]:


def outputshape(input_shape):
    return [(input_shape[0], input_shape[1]) for _ in range(input_shape[2])]

def normalise_pred(x):
    x = tf.stack(x)
    x = tf.transpose(x, [1, 0, 2])
    return x

def normalise_pred_shape(input_shape):
    shape = list(input_shape[0])
    num_trees = len(input_shape)
    return tuple([shape[0], num_trees, shape[1]])

    shape = list(input_shape[0])
    return tuple([shape[0], 2])


# In[7]:


def softmax_tau(proba, tau=0.1):
    """
    This is a softmax which goes towards one-hot encoding overtime. 
    We want to decay tau from 1.0 to 0.1 roughly
    """
    from scipy.special import logit, expit
    out = expit(logit(proba)/tau)
    return out/np.sum(out)

def get_layer_weights(model, name='hwy', sample=False, tau=1.0):
    out = K.eval([x for x in model.layers if x.name == name][0].weights[0]).flatten()
    return normalise_weights(out, sample, tau)

def normalise_weights(out, sample=False, tau=1.0):
    out = np.abs(out)
    out = out/np.sum(out)
    if sample and tau >= 1.0:
        draw = np.random.choice(range(out.shape[0]), 1, p=out)
        return draw[0]
    elif sample:
        draw = np.random.choice(range(out.shape[0]), 1, p=softmax_tau(out, tau))
        return draw[0]
    elif tau >= 1.0:
        return out
    else:
        return softmax_tau(out, tau)


# In[8]:


def calculate_routes(adj_list=None):
    """
    Calculates routes given a provided adjancency list,
    assume that root node is always 0.
    Assume this is a binary tree as well...
    Test cases:
    {0:[1, 2], 1:[], 2:[]} --> [(0, 0), (1, 0),
                                (0, 0), (1, 1),
                                (0, 1), (2, 0),
                                (0, 1), (2, 1)]
    {0:[1], 1:[2], 2:[]}   --> [(0, 0), (1, 0), (2, 0),
                                (0, 0), (1, 0), (2, 1),
                                (0, 0), (1, 1),
                                (0, 1)]
    calculate_routes({0:[1,2], 1:[], 2:[]})
    calculate_routes({0:[1], 1:[2], 2:[]})
    """
    if adj_list is None:
        raise Exception("Adj_list cannot be none")

    def get_next(path):
        next_paths = adj_list[path[-1]]
        if len(next_paths) > 0:
            for p in next_paths:
                get_next(path + [p])
        else:
            all_paths.append(path)

    all_paths = []
    get_next([0])

    # convert paths to indices...
    path_indx = []
    for path in all_paths:
        cur_path = []
        for cur_node, nxt_node in zip(path, path[1:]+[None]):
            # print(cur_node, nxt_node)
            pos_dir = np.array(sorted(adj_list[cur_node]))
            pos_idx = np.argwhere(pos_dir==nxt_node).flatten().tolist()
            if len(pos_idx) > 0 and len(pos_dir) == 2:  # i.e. has 2 children
                cur_path.append((cur_node, pos_idx[0]))
            elif len(pos_idx) > 0 and len(pos_dir) == 1:  # i.e. has 1 child
                path_indx.append(cur_path + [(cur_node, 1)])  # then it will have a leaf!
                cur_path.append((cur_node, pos_idx[0]))
            elif nxt_node is not None:
                cur_path.append((cur_node, pos_dir.shape[0]))
            else:
                path_indx.append(cur_path + [(cur_node, 0)])
                path_indx.append(cur_path + [(cur_node, 1)])
    return path_indx


def build_tree(main_input, depth, tree_number=0, last_only=True):
    """
    Builds a single decision tree, returns all the specs needed to preserve tree state...
    """

    # main_input = Input(shape=(dim_size,), name='main_input')
    tree_nodes = DecisionTreeNode(depth=depth, name=f'decision_tree{tree_number}')(main_input)
    tree_route = DecisionTreeRouting(depth=depth, name=f'decision_route{tree_number}')([main_input, tree_nodes])
    
    leaf_layers = layers.Lambda(lambda x: [tf.squeeze(y) for y in tf.split(x, [1 for _ in range(K.int_shape(x)[2])], axis=2)], output_shape=outputshape)(tree_route)
    
    pred_layer_tree = [Dense(num_class, activation='softmax', name="t{}_tree_l{}".format(tree_number, idx))(x) for idx, x in enumerate(leaf_layers)]
    
    stack_pred = layers.Lambda(normalise_pred, output_shape=normalise_pred_shape)(pred_layer_tree)
    tree_d = DecisionPredRouting(depth=depth)([stack_pred, tree_nodes])

    if last_only:
        return [tree_d]
    else:
        return [tree_d], [tree_d]+pred_layer_tree


def normalise_pred2(x):
    x = tf.stack(x)
    x = tf.transpose(x, [1, 0, 2])
    cl = K.sum(x, axis=1)
    cl = cl/tf.norm(cl, ord=1, axis=1, keepdims=True)
    return cl


def normalise_pred_shape2(input_shape):
    shape = list(input_shape[0])
    return tuple([shape[0], num_class])


main_input = Input(shape=(dim_size,), name='main_input')
tree = []
out_list = []

for idx in range(ntree):
    if last_only:
        tree.append(build_tree(main_input, depth, idx, last_only))
    else:
        t_, out = build_tree(main_input, depth, idx, last_only)
        tree.append(t_)
        out_list.extend(out)

stack_pred = layers.Lambda(normalise_pred2, output_shape=normalise_pred_shape2)([x[0] for x in tree])

if last_only:
    outputs = [stack_pred]
else:
    outputs = [stack_pred] + out_list

model = Model(inputs=main_input, outputs=outputs)

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

time_cb = TimingCallback()
print("Running model with {} layers".format(len(model.layers)))
hist = model.fit([x], [y for _ in range(len(outputs))],
                 validation_data=([x_test], [y_test for _ in range(len(outputs))]),
                 epochs=nepochs, verbose=2, 
                 callbacks = [time_cb])
hist_df = pd.DataFrame(hist.history)
print(pd.DataFrame(hist.history).iloc[-1])
hist_df['times'] = time_cb.times[-hist_df.shape[0]:]

hist_df.to_csv('{}/benchmark_rf{}_lastonly{}_{}.csv'.format(save_dir, ntree, last_only, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), index=True)


