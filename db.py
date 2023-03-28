import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


gold_data = pd.read_csv("gld_price_data.csv")

gold_data.isnull().sum()
gold_data= gold_data.dropna()

X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)

regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train,Y_train)


pickle.dump(regressor, open('model.pkl', 'wb'))