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

imgId = 0
print({'label': labels[imgId]})
print({'size':np.size(train[imgId]), 'data': train[imgId]})

def plot_image(image):
    fig=plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.savefig('./img/test_digit.png')

plot_image(train[imgId].reshape((28,28)))

# clf = make_pipeline(StandardScaler(),
#                     SGDClassifier(max_iter=1000, tol=1e-3))
# clf.fit(X, Y)

