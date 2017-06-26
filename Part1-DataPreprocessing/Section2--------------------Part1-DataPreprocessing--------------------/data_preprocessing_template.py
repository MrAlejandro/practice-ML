import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

np.set_printoptions(threshold=np.nan)
path = os.path.dirname(os.path.abspath(__file__))

dataset = pd.read_csv(path + '/../Data.csv')
X = dataset.iloc[:, :-1].values # matrix of features (independent variables)
y = dataset.iloc[:, 3].values # dependent variables

# splitting the dataset to Training and Test stes
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# features scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""