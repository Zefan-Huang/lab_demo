import numpy as np

y = np.load('output/ready/y_train.npy')
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))
