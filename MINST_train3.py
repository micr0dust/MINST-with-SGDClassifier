"""
    訓練集做 cross-validation = 3，每次將剩下 2/3 的訓練集再向上下左右移
    動產生 4 張新圖，再開始 fit，輸出每次 predict 後的 confusion matrix 
    和 accuracy score，最後印出 3 次分數和平均。

    - 有除 255 正規化。
    - max_iter 原本是 1000，但因為會無法收斂就調 10000。
    - 輸出 confusion matrix 圖檔後的那串數字是訓練張數-驗證張數。
    - 會從 ./input/modified.csv 讀資料訓練，假設訓練集為 train，train[5*i] 都是原資
      料，train[5*i+1]、train[5*i+2]、train[5*i+3]、train[5*i+4] 分別為平移過下、上
      、右、左的資料。
    - 驗證集只取 train[5*i] 的原資料
"""
""" Output:
[0.91021429 0.9145     0.90828571]
0.911
"""
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

train_df = pd.read_csv("./input/modified.csv")
train_data = train_df.values

labels = train_data[:,0]
train = train_data[:,1:]/255

print(f'train_length: {len(train)}')

def plotConfMatric(true_labels, predicted_labels, fileName="confusion_matrix"):
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
    disp.plot()
    # plt.show()
    # print(fileName)
    plt.savefig(f'img/{fileName}.png')

def my_cross_val_score(model, train, label, cv=3, scoring='accuracy'):
    N = len(train)//5
    part = N//cv
    indices = [i * part for i in range(3)]
    indices = np.append(indices, N)
    scores = []
    for i in range(cv):
        print(f'cv{i+1} selecting train data...')
        cv_label = np.array([])
        arr1,arr2 = train[0:5*indices[i]], train[5*indices[i+1]:5*N]
        label1,label2 = label[0:5*indices[i]], label[5*indices[i+1]:5*N]
        cv_train = np.concatenate((arr1,arr2))
        cv_label = np.concatenate((label1,label2))
        print(f'cv{i+1} fitting...')
        model.fit(cv_train, cv_label)
        print(f'cv{i+1} predicting...')
        valid = train[5*indices[i]:5*indices[i+1]:5]
        answer = label[5*indices[i]:5*indices[i+1]:5]
        predictions = clf.predict(valid)
        score = accuracy_score(answer, predictions)
        plotConfMatric(answer, predictions, f"train3_{i+1}_{len(cv_train)}-{len(predictions)}")
        print(f'score: {score}')
        scores.append(score)
    return np.array(scores)


clf = make_pipeline(SGDClassifier(max_iter=100000))

scores = my_cross_val_score(clf, train, labels, cv=3, scoring='accuracy')

print(scores)
print(scores.mean())