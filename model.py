import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR

import mlflow
import mlflow.sklearn

SRC_PATH = './data/raw/'

# *******************************************************************
# Data Loading
# *******************************************************************
def data_load(data_file):
    """
    Transform the source csv data into pandas DataFrame and fill the empty rows with median value.
    """
    df = pd.read_csv(os.path.join(SRC_PATH, data_file))
    return df

def clean_data(df):
    df = df.fillna(df.mean())

    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Find features with correlation greater than 0.9
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

    # Drop features 
    df.drop(to_drop, axis=1, inplace=True)
    return df

# *******************************************************************
# Split Target and Features (Un-preprocessed)
# *******************************************************************
def target_features_split(target_name, feat_list, df):
    """
    Split the DataFrame into two DataFrames: one is for the target, and the
    other is for the features(un-preprocessed).

    This gives flexibilty to select specific target columns only.

    """
    target = df[target_name].to_frame()
    features = df[feat_list]
    return target, features

# *******************************************************************
# Split the Training and Testing Data
# *******************************************************************
def split(features, target, random_state=0):
    """
    Split the data into training and testing set.
    """
    X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=1/3,random_state=0)
    return X_train, X_test, y_train, y_test



# *******************************************************************
# Build the Model
# *******************************************************************
def regressor_model(modelName):
    """
    Build the regressor tree model.
    """
    if modelName == 'Linear Regression':
        model = LinearRegression()
    elif modelName == 'LGBM':
        model = LGBMRegressor()
    elif modelName == 'XGB':
        model = XGBRegressor()
    elif modelName == 'SGD':
        model = SGDRegressor()
    elif modelName == 'Elastic Net':
        model = ElasticNet()
    elif modelName == 'Bayesian':
        model = BayesianRidge()
    elif modelName == 'SVR':
        model = SVR()
    else:
        print("Give valid model name!")
    return model


def train():
    df = data_load('220527_seqana_mlops_challenge_dataset.csv')

    X = df.iloc[:,1:-1]
    y = (df.iloc[:,-1:])
    # print ("Non clean X shape: ", X,y)

    X_clean = clean_data(X)
    # print("clean X shape", X_clean.shape)
    X_train, X_test, y_train, y_test = split(X_clean,y)


    with mlflow.start_run():
        modelName = 'Elastic Net'
        model = regressor_model(modelName)
    
        reg = model.fit(X_train, y_train)
    
        y_pred = model.predict(X_test)
        
        print("MAE score: ", mean_absolute_error(y_test, y_pred))
        print('Training R2 score: ' + str(reg.score(X_train,y_train)))
        print('Test R2 score: ' + str(reg.score(X_test,y_test)))
        mlflow.log_param("Model name", modelName)
        mlflow.log_metric("R2 score", reg.score(X_test,y_test))
        mlflow.log_metric("MAE score", mean_absolute_error(y_test, y_pred))


if __name__ == '__main__':
    train()