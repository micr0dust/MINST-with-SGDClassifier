"""
    訓練集做 cross-validation = 3 ， 印出3次分數和平均。

    - 有除 255 正規化。
    - max_iter 原本是 1000，但因為會無法收斂就調 10000。
"""
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv("./input/train.csv")
train_data = train_df.values

labels = train_data[:,0]
train = train_data[:,1:]/255


clf = make_pipeline(SGDClassifier(max_iter=100000))

scores = cross_val_score(clf, train, labels, cv=3, scoring='accuracy')
print(scores)
print(scores.mean())