"""
    把訓練集前 5000 筆當驗證集，用 SGDClassifier fit 過後，輸出 predict 後
    的 confusion matrix 和 accuracy score。

    - 有除 255 正規化。
    - max_iter 原本是 1000，但因為會無法收斂就調 10000。
    - 輸出 confusion matrix 圖檔後的那串數字是訓練張數-驗證張數。
"""
""" Output
score: 0.913
"""
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plotConfMatric(true_labels, predicted_labels, fileName="confusion_matrix"):
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
    disp.plot()
    # plt.show()
    plt.savefig(f'img/{fileName}.png')

train_df = pd.read_csv("./input/train.csv")
train_data = train_df.values

labels = train_data[:,0]
train = train_data[:,1:]/255
train, valid = train[5000:], train[:5000]
labels, answer = labels[5000:], labels[:5000]

clf = make_pipeline(SGDClassifier(max_iter=100000))
clf.fit(train, labels)
predictions = clf.predict(valid)
score = accuracy_score(answer, predictions)
plotConfMatric(answer, predictions, f"train0_{len(train)}-{len(predictions)}")
print(f"score: {score}")