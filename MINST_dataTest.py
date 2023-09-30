"""
    測試資料集用
"""
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import shift


train_df = pd.read_csv("./input/train.csv")
train_data = train_df.values

labels = train_data[:,0]
train = train_data[:,1:]

print(f"train size: {len(train)}")

def plot_image(image, fileName="test_digit"):
    fig=plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.savefig(f'./img/{fileName}.png')

def getImg(a, b):
    for i in range(a,b):
        plot_image(train[i].reshape((28,28)), f"test_digit{i}")

def imgInfo(idx):
    print({'label': labels[idx]})
    print({'size':np.size(train[idx]), 'data': train[idx]})

# getImg(imgId, imgId)    

# plot_image(shift(train[imgId].reshape((28,28)), [0,10], cval=0), "test_digit2")
# clf = make_pipeline(StandardScaler(),
#                     SGDClassifier(max_iter=1000, tol=1e-3))
# clf.fit(X, Y)

