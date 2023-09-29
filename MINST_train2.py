"""
    訓練集做 cross-validation = 3，每次將剩下 2/3 的訓練集再向上下左右移
    動產生 4 張新圖，再開始 fit，輸出每次 predict 後的 confusion matrix 
    和 accuracy score，最後印出 3 次分數和平均。

    - 有除 255 正規化。
    - max_iter 原本是 1000，但因為會無法收斂就調 10000。
    - 輸出 confusion matrix 圖檔後的那串數字是訓練張數-驗證張數。
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

train_df = pd.read_csv("./input/train.csv")
train_data = train_df.values

labels = train_data[:,0]
train = train_data[:,1:]/255

print(f'train_length: {len(train)}')

def plotConfMatric(true_labels, predicted_labels, fileName="confusion_matrix"):
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
    disp.plot()
    # plt.show()
    plt.savefig(f'img/{fileName}.png')

def modify(data, label):
    actions = [[1,0],[-1,0],[0,1],[0,-1]]
    res_data = np.array(data)
    res_label = np.array(label)
    N = len(data)
    for i in range(N):
        if i%(int(N/10))==0:
            print(str(int(i/N*100))+'%')
        for action in actions:
            modified = shift(data[i].reshape((28,28)), action, cval=0)
            res_data=np.append(res_data, modified.reshape((1,-1)))
        res_label=np.append(res_label, [label[i] for j in range(len(actions))])
    return res_data, res_label


def my_cross_val_score(model, train, label, modify, cv=1, scoring='accuracy'):
    scores = np.array([])
    train = np.array([np.array(train[i::cv]) for i in range(cv)])
    label = np.array([np.array(label[i::cv]) for i in range(cv)])
    for i in range(cv):
        cv_train = np.array([])
        cv_label = np.array([])
        process = 0
        for j in range(cv):
            if j!=i:
                process+=1
                print(f'cv-{i+1} process:{process}/{cv-1}')
                m_train, m_label=modify(np.array(train[j]),np.array(label[j]))
                cv_train=np.append(cv_train, m_train)
                cv_label=np.append(cv_label, m_label)
        cv_train=cv_train.reshape((-1,784))
        print(f'cv{i+1} fitting...')
        model.fit(cv_train, cv_label)
        print(f'cv{i+1} predicting...')
        predictions = clf.predict(train[i])
        score = accuracy_score(label[i], predictions)
        plotConfMatric(label[i], predictions, f"train2_{i+1}_{len(cv_train)}-{len(predictions)}")
        print(f'score: {score}')
        scores = np.append(scores, score)
    return scores


clf = make_pipeline(SGDClassifier(max_iter=100000))

scores = my_cross_val_score(clf, train, labels, modify, cv=3, scoring='accuracy')

print(scores)
print(scores.mean())