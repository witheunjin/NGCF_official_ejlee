# NGCF_official_ejlee
This repository is from `xiangwang1223/neural_graph_collaborative_filtering`.

### Related Materials
* Paper: [Neural Graph Collaborative Filtering](https://arxiv.org/pdf/1905.08108.pdf)
* Repository: [xiangwang1223/neural_graph_collaborative_filtering](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)

INITIAL EXECUTION

```
2021-06-23 15:49:18.205364: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
n_users=29858, n_items=40981
n_interactions=1027370
n_train=810128, n_test=217242, sparsity=0.00084
already create adjacency matrix (70839, 70839) 47.68931555747986
generate single-normalized adjacency matrix.
generate single-normalized adjacency matrix.
already normalize adjacency matrix 1.3775660991668701
use the normalized adjacency matrix
Traceback (most recent call last):
  File "NGCF.py", line 360, in <module>
    model = NGCF(data_config=config, pretrain_data=pretrain_data)
  File "NGCF.py", line 53, in __init__
    self.users = tf.placeholder(tf.int32, shape=(None,))
AttributeError: module 'tensorflow' has no attribute 'placeholder'
```
SOLUTION: Install 'placeholder'
```
$ sudo pip install placeholder
```
RESULT
```
Collecting placeholder
  Downloading placeholder-1.2.1-cp38-cp38-manylinux2014_x86_64.whl (20 kB)
Installing collected packages: placeholder
Successfully installed placeholder-1.2.1
```
Put this code
```python3
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

NEXT ERROR MESSAGE
```
2021-06-23 15:55:50.771415: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
n_users=29858, n_items=40981
n_interactions=1027370
n_train=810128, n_test=217242, sparsity=0.00084
already load adj matrix (70839, 70839) 0.13281631469726562
use the normalized adjacency matrix
WARNING:tensorflow:From /home/ygkim/.local/lib/python3.8/site-packages/tensorflow/python/compat/v2_compat.py:96: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
Traceback (most recent call last):
  File "NGCF.py", line 362, in <module>
    model = NGCF(data_config=config, pretrain_data=pretrain_data)
  File "NGCF.py", line 72, in __init__
    self.weights = self._init_weights()
  File "NGCF.py", line 119, in _init_weights
    initializer = tf.contrib.layers.xavier_initializer()
AttributeError: module 'tensorflow' has no attribute 'contrib'
```
