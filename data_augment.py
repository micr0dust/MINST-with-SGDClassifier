import numpy as np
import pandas as pd
from scipy.ndimage import shift

train_df = pd.read_csv("./input/train.csv")
train_data = train_df.values

labels = train_data[:,0]
train = train_data[:,1:]

csv_file = './input/modified.csv'
batch_size = 1000
def augment_and_write_batch(labels, train, csv_file, mode):
    actions = [[1,0],[-1,0],[0,1],[0,-1]]
    res_data=[]
    for img in train:
        res_data.append(img)
        original_image = img.reshape((28, 28))
        for action in actions:
            modified = shift(original_image, action, cval=0)
            res_data.append(modified.flatten())

    df_batch = pd.DataFrame(data=res_data, columns=[f'pixel{i}' for i in range(len(res_data[0]))])
    labels_col = [label for label in labels for _ in range(5)]
    df_batch.insert(0, 'label', labels_col)

    df_batch.to_csv(csv_file, mode=mode, index=False, header=(mode == 'w'))

    
N = len(labels)
for i in range(0, N, batch_size):
    print(str(int(i/N*100))+'%')
    labels_batch = labels[i:i + batch_size]
    train_batch = train[i:i + batch_size]
    mode = 'w' if i == 0 else 'a'
    augment_and_write_batch(labels_batch, train_batch, csv_file, mode)

print(f'success write to {csv_file}')
