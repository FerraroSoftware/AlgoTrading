# Distribution functions
from scipy.stats import norm


# Load libraries
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

#Libraries for Deep Learning Models
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasRegressor

#Libraries for Statistical Models
import statsmodels.api as sm

#Libraries for Saving the Model
from pickle import dump
from pickle import load

# Time series Models
from statsmodels.tsa.arima_model import ARIMA
#from statsmodels.tsa.statespace.sarimax import SARIMAX

# Error Metrics
from sklearn.metrics import mean_squared_error

# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression


#Plotting 
from pandas.plotting import scatter_matrix
from statsmodels.graphics.tsaplots import plot_acf

#Diable the warnings
import warnings
warnings.filterwarnings('ignore')


class Deriv():
   def __init__(alpha, beta, sigma, risk_free_rate) -> None:
       self.alpha = alpha
       self.beta = beta
       self.sigma = sigma
       self.risk_free_rate = risk_free_rate


   def option_vol_from_surface(self, moneyness, time_to_maturity):
      return self.sigma + self.alpha * time_to_maturity + self.beta * np.square(moneyness - 1)

   def call_option_price(moneyness, time_to_maturity, option_vol):
      d1=(np.log(1/moneyness)+(self.risk_free_rate + np.square(option_vol))*time_to_maturity)/(option_vol*np.sqrt(time_to_maturity))

      d2=(np.log(1/moneyness)+(self.risk_free_rate - np.square(option_vol))*time_to_maturity)/(option_vol*np.sqrt(time_to_maturity))
      
      N_d1 = norm.cdf(d1)
      N_d2 = norm.cdf(d2)

      return N_d1 - moneyness * np.exp(-self.risk_free_rate*time_to_maturity) * N_d2