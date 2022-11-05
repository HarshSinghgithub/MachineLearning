import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 

#Read data from csv file
data = pd.read_csv('train.csv')

#Remove unuseful columns
col = ['Name', 'Ticket', 'Cabin']
data = data.drop(col, axis=1)

data = data.dropna()

data.info()

#Convert data frame to numpy
X = data.values
X = np.delete(X, 1, axis=1)
Y = data['Survived'].values

X_Train , X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.7)

