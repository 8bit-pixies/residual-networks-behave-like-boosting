README
======

Here is the notebooks and code used to run experiments used on the paper "Residual Networks Behave Like Boosting Algorithms" which is part of the proceedings of 2019 IEEE DSAA conference. 

This repository will not be maintained. Work on this paper and related algorithms will be moving to this repository in the future: https://github.com/chappers/treegrad

```
@inproceedings{siu2019residual,
  title={Residual Networks Behave Like Boosting Algorithms},
  author={Siu, Chapman},
  booktitle={2019 {IEEE} International Conference on Data Science and Advanced Analytics, DSAA 2019, Washington DC, USA, October 5-8, 2019},
  year={2019},
  organization={IEEE}
}
```

cifar10 notebooks should be runnable as is. SVHN requires appropriate data. 

```
python resnet_dt.py
```

Will run over a dummy mandelon dataset and should reach 100% accuracy with parameters provided (which are variant of the ones used in the paper)

directory
---------

Folder: code_dump has the code for all datasets 
clean_data: is the datasets used with splits


Environment Package List
========================

```
Package             Version
------------------- ---------
absl-py             0.2.2
anaconda-client     1.6.14
asn1crypto          0.24.0
astor               0.6.2
astroid             2.0.4
atomicwrites        1.2.1
attrs               18.2.0
backcall            0.1.0
bleach              1.5.0
certifi             2018.8.24
cffi                1.11.5
chardet             3.0.4
cloudpickle         0.6.1
clyent              1.2.2
colorama            0.3.9
coverage            4.5.2
cryptography        2.2.2
cycler              0.10.0
dask                1.0.0
decorator           4.3.0
entrypoints         0.2.3
flake8              3.5.0
gast                0.2.0
grpcio              1.12.1
h5py                2.8.0
html5lib            0.9999999
idna                2.7
ipykernel           4.8.2
ipython             6.4.0
ipython-genutils    0.2.0
ipywidgets          7.2.1
isort               4.3.4
jedi                0.12.0
Jinja2              2.10
jsonschema          2.6.0
jupyter-client      5.2.3
jupyter-core        4.4.0
jupyterlab          0.35.0
jupyterlab-launcher 0.10.5
jupyterlab-server   0.2.0
Keras               2.2.0
Keras-Applications  1.0.6
Keras-Preprocessing 1.0.5
kiwisolver          1.0.1
lazy-object-proxy   1.3.1
lightgbm            2.2.1
lime                0.1.1.32
Markdown            2.6.11
MarkupSafe          1.0
matplotlib          3.0.0
mccabe              0.6.1
mistune             0.8.3
mkl-fft             1.0.0
mkl-random          1.0.1
more-itertools      4.3.0
nb-anacondacloud    1.4.0
nb-conda            2.2.0
nb-conda-kernels    2.1.0
nbconvert           5.3.1
nbformat            4.4.0
nbpresent           3.0.2
networkx            2.2
nose                1.3.7
nose2               0.8.0
notebook            5.5.0
numpy               1.14.5
pandas              0.23.1
pandocfilters       1.4.2
parso               0.2.1
pickleshare         0.7.4
Pillow              5.3.0
pip                 10.0.1
pkginfo             1.4.2
plotly              3.4.2
pluggy              0.8.0
prompt-toolkit      1.0.15
protobuf            3.6.1
py                  1.7.0
py4j                0.10.7
pycodestyle         2.3.1
pycparser           2.18
pyflakes            1.6.0
Pygments            2.2.0
pylint              2.2.2
pyOpenSSL           18.0.0
pyparsing           2.2.1
PySocks             1.6.8
pyspark             2.3.1
pytest              4.0.2
python-dateutil     2.7.3
pytz                2018.5
PyWavelets          1.0.1
pywinpty            0.5.4
PyYAML              3.12
pyzmq               17.0.0
requests            2.19.1
requests-toolbelt   0.8.0
retrying            1.3.3
scikit-image        0.14.1
scikit-learn        0.20.2
scikit-optimize     0.5.2
scipy               1.1.0
Send2Trash          1.5.0
setuptools          40.0.0
simplegeneric       0.8.1
six                 1.11.0
sklearn-pandas      1.5.0
sklearn2pmml        0.24.1
skope-rules         1.0.0
SQLAlchemy          1.2.15
tensorboard         1.12.1
tensorflow          1.12.0
termcolor           1.1.0
terminado           0.8.1
testpath            0.3.1
toolz               0.9.0
torch               0.4.1
torchsummary        1.5.1
torchvision         0.2.1
tornado             5.0.2
tqdm                4.23.4
traitlets           4.3.2
twine               1.11.0
typed-ast           1.1.0
urllib3             1.23
wcwidth             0.1.7
Werkzeug            0.14.1
wheel               0.31.1
widgetsnbextension  3.2.1
win-inet-pton       1.0.1
wincertstore        0.2
wrapt               1.10.11
```



