"""
    模擬 cross-validation + data augmentation 用
"""
import numpy as np

cv = 3
train = np.array([1,2,3,4,5,6,7,8,9])
label = np.array([1,2,3,4,5,6,7,8,9])
train = np.array([np.array(train[i::cv]) for i in range(cv)])
label = np.array([np.array(label[i::cv]) for i in range(cv)])
print(train)

def shift(arr):
    for i in range(np.size(arr)):
        arr[i] = -arr[i]
    return arr

for i in range(cv):
    cv_train = np.array([])
    cv_label = np.array([])
    for j in range(cv):
        if j!=i:
            cv_train=np.append(cv_train, train[j])
            cv_label=np.append(cv_label, label[j])
            cv_train=np.append(cv_train, shift(np.array(train[j])))
            cv_label=np.append(cv_label, label[j])
    print({
        'train': cv_train,
        'train_label': cv_label,
        'valid': train[i],
        'valid_label': label[i],
    })
