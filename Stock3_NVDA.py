# Importing the required Python Libraries to handle the data
import numpy as np
import pandas as pd

# Importing the required metrics to calculate statistics
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

# Import the dataset
data = pd.read_csv("Data/NVDA.csv")

# Separting Features from the Target
X = data[['Open','High','Low', 'Close']].to_numpy()
Y = data['Adj Close'].to_numpy()

# Data Normalization using MinMax Scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_x_data = scaler.fit_transform(X)
scaled_y_data = scaler.fit_transform((Y).reshape(-1, 1))

# Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(scaled_x_data, scaled_y_data, test_size=0.2)




""" Random Forest Regressor Algorithm """

# Importing the Algorithm from the library
from  sklearn.ensemble import RandomForestRegressor
random_forest_reg = RandomForestRegressor()

# Fitting the model to the training dataset
random_forest_reg.fit(X_train, Y_train)

print()
print('RF Reg Train Score: ',random_forest_reg.score(X_train,Y_train))
print('RF Reg Test Score: ',random_forest_reg.score(X_test,Y_test))
print()

# Predicting the Test Results
Y_pred_rf = random_forest_reg.predict(X_test)

# Calcukate the three metrics for Random Forest Regressor

# Calculate the Mean Absolute Error 
rf_mae = mae(Y_test, Y_pred_rf)

# Calculate the Mean Square Error
rf_mse = mse(Y_test, Y_pred_rf)

# Calculate the r2 Score
rf_r2 = r2(Y_test, Y_pred_rf)




""" Decision Tree Regressor Algorithm """

# Applying the Decision Tree Regressor Algorithm
from sklearn.tree import DecisionTreeRegressor
decision_tree_reg = DecisionTreeRegressor()

# Fitting the Decision Tree Regressor model to the training dataset
decision_tree_reg.fit(X_train,Y_train)

print()
print('DT Reg Train Score: ',decision_tree_reg.score(X_train,Y_train))
print('DT Reg Test Score: ',decision_tree_reg.score(X_test,Y_test))
print()

# Predicting the Test Results
Y_pred_dt = decision_tree_reg.predict(X_test)

# Calcukate the three metrics for Decison Tree Regressor

# Calculate the Mean Absolute Error 
dt_mae = mae(Y_test, Y_pred_dt)

# Calculate the Mean Square Error
dt_mse = mse(Y_test, Y_pred_dt)

# Calculate the r2 Score
dt_r2 = r2(Y_test, Y_pred_dt)




""" K-Nearest Neighbor Regressor Algorithm """

# Applying K-Nearest Neighbor Regressor Algorithm
from sklearn.neighbors import KNeighborsRegressor
KNN_reg = KNeighborsRegressor(n_neighbors = 1)

# Fitting the KNN Regressor model to the training dataset
KNN_reg.fit(X_train, Y_train)

print()
print('KNN Reg Train Score: ',KNN_reg.score(X_train,Y_train))
print('KNN Reg Test Score: ',KNN_reg.score(X_test,Y_test))
print()

# Predicting the Test Results
Y_pred_knn = KNN_reg.predict(X_test)

# Calcukate the three metrics for K-Nearest Neighbor Regressor

# Calculate the Mean Absolute Error 
knn_mae = mae(Y_test, Y_pred_knn)

# Calculate the Mean Square Error
knn_mse = mse(Y_test, Y_pred_knn)

# Calculate the r2 Score
knn_r2 = r2(Y_test, Y_pred_knn)




""" Support Vector Machines Regressor Algorithm """ 

from sklearn.svm import SVR
SVM_reg = SVR(kernel = 'rbf')

# Fitting the Support Vector Regressor Algorithm
SVM_reg.fit(X_train, Y_train)

print()
print('SVM Reg Train Score: ',SVM_reg.score(X_train,Y_train))
print('SVM Reg Test Score: ',SVM_reg.score(X_test,Y_test))
print()

# Predicting the Test Results
Y_pred_svm = SVM_reg.predict(X_test)

# Calcukate the three metrics for K-Nearest Neighbor Regressor

# Calculate the Mean Absolute Error 
svm_mae = mae(Y_test, Y_pred_svm)

# Calculate the Mean Square Error
svm_mse = mse(Y_test, Y_pred_svm)

# Calculate the r2 Score
svm_r2 = r2(Y_test, Y_pred_svm)
