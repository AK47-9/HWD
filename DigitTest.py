import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv("train.csv").values
clf = DecisionTreeClassifier()

# training dataset
xtrain = data[0:21000,1:]
train_label = data[0:21000,0]

clf.fit(xtrain,train_label)

#testing data
xtest = data[21000: ,1:]
actual_label = data[21000: ,0]

#testing
d = xtest[9]

d.shape = (28,28)
pt.imshow(d,cmap='gray')
print(clf.predict( [xtest[9]]))
pt.show()
