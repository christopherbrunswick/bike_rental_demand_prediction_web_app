#!/usr/bin/env python
# coding: utf-8

# # Client-Side Intelligence Using Regression Coefficients on AWS CLI (Monetizing Machine Learning Project)
# 
# - Interactive web application to understand bike rental demand using regression coefficients on Amazon Web Services
# 
# In this project we'll model the Bike Sharing Dataset from Capital Bikeshare System using a regression model to analysis how temperature, wind, and time affect bicycle rentals in the mid-atlantic region of the US.
# 
# The primary purpose of regression as a supervised learning task is to compose an adequate model to predict the continuous dependent attributes from many of the independent random variables (predictors). Specifically, regression is a statistical measurement that defines the magnitute of the relationship between the target variable (outcome or dependent variable) and one or more independent variables.
# 
# ### Features
# - instant: record index
# - dteday : date
# - season : season (1:springer, 2:summer, 3:fall, 4:winter)
# - yr : year (0: 2011, 1:2012)
# - mnth : month ( 1 to 12)
# - hr : hour (0 to 23)
# - holiday : weather day is holiday or not (extracted from [Web Link])
# - weekday : day of the week
# - workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
# - weathersit :
#     1: Clear, Few clouds, Partly cloudy, Partly cloudy
#     2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#     3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#     4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
#     
# - temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in     hourly scale)
# - atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50      (only in hourly scale)
# - hum: Normalized humidity. The values are divided to 100 (max)
#     windspeed: Normalized wind speed. The values are divided to 67 (max)
# - windspeed: Normalized wind speed. The values are divided to 67 (max)
# - casual: count of casual users
# - registered: count of registered users
# - cnt: count of total rental bikes including both casual and registered

# ## Preliminary Packages

# In[1]:


#web scraping packages to acquire data
import requests
from bs4 import BeautifulSoup

#basic packages
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

#scikit learn packages
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

from zipfile import ZipFile

#functions
def plot_learning_curves(model, X_train, y_train):
    train_errors, test_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predictions = model.predict(X_train[:m])
        y_test_predictions = model.predict(X_test)
        train_errors.append(np.sqrt(mean_squared_error(y_train[:m], y_train_predictions)))
        test_errors.append(np.sqrt(mean_squared_error(y_test, y_test_predictions)))
    plt.plot(train_errors, "r-+", linewidth=2, label="training curve")
    plt.plot(test_errors, "b-", linewidth=3, label="testing or validation curve")


# ## Scrape Website For Data Acquisition Using BeautifulSoup

# In[5]:


import requests
from bs4 import BeautifulSoup

# URL of the UCI Machine Learning Repository page
url = "https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset"

# Send an HTTP GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the page with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the download link with the specified class and attribute
    download_link = soup.find('a', class_='btn-primary', download='bike+sharing+dataset.zip')

    # Check if the download link was found
    if download_link:
        # Extract the href attribute, which contains the download URL
        dataset_url = "https://archive.ics.uci.edu" + download_link.get('href')

        # Specify the local file path where you want to save the dataset
        local_file_path = r"C:/Users/cbrun/Documents/portfolio_project_2/bike_sharing_dataset.zip"

        # Download the dataset and save it to the local file path
        response = requests.get(dataset_url)
        with open(local_file_path, 'wb') as file:
            file.write(response.content)

        print(f"Dataset downloaded and saved to {local_file_path}")
    else:
        print("Download link not found on the page.")
else:
    print("Failed to retrieve the page. Status code:", response.status_code)


# In[9]:


file = 'C:/Users/cbrun/Documents/portfolio_project_2/bike_sharing_dataset.zip'
with ZipFile(file, 'r') as zfile:
    zfile.extractall('/portfolio_project_2/bike_sharing_dataset')


# In[2]:


bikes_hour_df_raw = pd.read_csv('/portfolio_project_2/bike_sharing_dataset/hour.csv')
bikes_day_df_raw = pd.read_csv('/portfolio_project_2/bike_sharing_dataset/day.csv')


# ## Exploring Data

# In[4]:


bikes_hour_df_raw.head(10)


# In[3]:


#keeping features that are needed for the web application - the features dropped are not significant in modeling demand from
#a single user's perspective according to the text
bikes_hour_df = bikes_hour_df_raw.drop(columns=['casual', 'registered'], axis=1)


# In[4]:


bikes_hour_df = bikes_hour_df.rename(columns={'cnt':'count'})


# In[7]:


print(bikes_hour_df.describe())


# In[8]:


bikes_hour_df.info()


# In[9]:


bikes_hour_df.isnull().sum()


# In[10]:


bikes_hour_df.duplicated().sum()


# The <strong>dteday</strong> feature is text data which is irrelevant for our web application and could possibly impact the accuracy of our model. However, we will bypass dropping the feature at this point. Also, there are no missing data points in this dataset per the Non-Null column from info(). Additionally, there are no duplicated rows within this dataframe.

# In[53]:


bikes_hour_df['count'].describe()


# ## Visualization of Numerical Features

# In[9]:


fig, ax = plt.subplots(1)
ax.plot(sorted(bikes_hour_df['count']), color='brown')
ax.set_xlabel("Row Index", fontsize=12)
ax.set_ylabel("Sorted Rental Counts", fontsize=12)
fig.suptitle('Outcome Variable - cnt - Rental Counts')
plt.show()


# In[12]:


fig, ax = plt.subplots(1)
ax.scatter(bikes_hour_df['temp'], bikes_hour_df['count'], color='green')
ax.set_xlabel("temp", fontsize=12)
ax.set_ylabel("Count of All Bikes Rented", fontsize=12)
fig.suptitle('Numerical Feature: Cnt vs temp')
plt.show()


# In[14]:


fig, ax = plt.subplots(1)
ax.scatter(bikes_hour_df['atemp'], bikes_hour_df['count'], color='red')
ax.set_xlabel("atemp", fontsize=12)
ax.set_ylabel("Count of All Bikes Rented", fontsize=12)
fig.suptitle('Numerical Feature: Cnt vs atemp')
plt.show()


# The visualization shows that there is a linear connection between the number of bikes rented and temperature, however the similarity in the distribution of both temp and atemp may cause collinearity between the two features or independent variable. This can cause our model to underperform.

# In[15]:


fig, ax = plt.subplots(1)
ax.scatter(bikes_hour_df['hum'], bikes_hour_df['count'], color='purple')
ax.set_xlabel("hum", fontsize=12)
ax.set_ylabel("Count of All Bikes Rented", fontsize=12)
fig.suptitle('Numerical Feature: Cnt vs hum')
plt.show()


# In[16]:


fig, ax = plt.subplots(1)
ax.scatter(bikes_hour_df['windspeed'], bikes_hour_df['count'], color='orange')
ax.set_xlabel("windspeed", fontsize=12)
ax.set_ylabel("Count of All Bikes Rented", fontsize=12)
fig.suptitle('Numerical Feature: Cnt vs windspeed')
plt.show()


# In[58]:


sns.histplot(bikes_hour_df, x="count")
sns.despine()
plt.title("Bike Count")
plt.show()


# ## Visualization of Categorical Features

# In[25]:


fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,10))
ax1 = bikes_hour_df[['season','cnt']].groupby(['season']).sum().reset_index().plot(kind='bar',
                                       legend = False, title ="Counts of Bike Rentals by season", 
                                         stacked=True, fontsize=12, ax=ax1)
ax1.set_xlabel("season", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)
ax1.set_xticklabels(['spring', 'summer', 'fall', 'winter'])



ax2 = bikes_hour_df[['weathersit','cnt']].groupby(['weathersit']).sum().reset_index().plot(kind='bar',  
      legend = False, stacked=True, title ="Counts of Bike Rentals by weathersit", fontsize=12, ax=ax2)
ax2.set_xlabel("weathersit", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)
ax2.set_xticklabels(['1: Clear', '2: Mist', '3: Light Snow', '4: Heavy Rain'])

fig.suptitle('Count of Bike Rentals by weathersit')
plt.show()


# In[26]:


# alternative way of plotting using groupby
ax = bikes_hour_df[['hr','cnt']].groupby(['hr']).sum().reset_index().plot(kind='bar', figsize=(8, 6),
                                       legend = False, title ="Total Bike Rentals by Hour", 
                                       color='orange', fontsize=12)
ax.set_xlabel("Hour", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
plt.show()


# ## Modeling
# 
# The first model of choice is <strong>linear regression (multilinear regression)</strong> where we are looking to make a prediction on the continuous variable (quantitative variable) count. Our independent variables may be either continous or discrete (categorical or qualitative variables), however the discrete variables must have unique data points to encode.

# ### Linear Regression 

# In[5]:


# simple approach - make a copy for editing without affecting original
bike_df_model_ready = bikes_hour_df.copy()

target = 'count'

# create a feature list for each modeling - experiment by adding features to the exclusion list
features = [feat for feat in list(bike_df_model_ready) if feat not in [target, 'instant', 'dteday']]  

# split data into train and test portions and model
X_train, X_test, y_train, y_test = train_test_split(bike_df_model_ready[features], 
                                                 bike_df_model_ready[['count']], 
                                                 test_size=0.2, random_state=42)


from sklearn import linear_model

# instantiate LinearRegression() object
reg = linear_model.LinearRegression()
 
# train the model on training set
reg.fit(X_train, y_train) 

# check model accuracy score
print("Adjusted R^2: %.2f" % (1 - (1-reg.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)))

# make predictions using the testing set
predictions = reg.predict(X_test)

# root mean squared error to assess how well our model fits the dataset
print("Root Mean squared error: %.2f" % sqrt(mean_squared_error(y_test, predictions)))


# In[35]:


plot_learning_curves(reg, X_train, y_train)


# Our Linear Regression tells us that the average deviation between the predicted data points scored and the actual data points scored is 139.21. According to the text, another way to interpret or evaluate the accuracy score of our model is to look at the mean value of our target variable. Since our metric is on the same scale of our target variable, we can say that our model predictions are off by 139 bikes. As the mean bike rental demand per hour is approximately 190, our model does a better job than what the sample mean of bike rentals dictates. We can say this because 0 error between our predictions and original data points is ideal but highly unlikely unless our model is overfitting. We are not sure if this is the best model so we will build an additional model to see if we get a lower RMSE. Futhermore, the model is somewhat of a mediocre model since it only explains 39% of the variation in our outcome variable, in other words we can expect 39% of the variation in our response or outcome variable to be explained by our linear regression model. (Only 39% of the data points fall within the regression line) We are looking to get as close to 100% as possible

# The above code uses train_test_split() function to split the training set into a subset of the training set and a validation set. We then trained the linear regression model against the subset training set and evaluated it against the validation set. We will now use an alternative to train_test_split() which is the k-fold cross-validation feature offered in scikit learn. KFold will randomly split the training set into 10 distinct subsets called folds. The cross_val_score feature will then train and evaluate the linear regression model 10 times, picking a different fold for evaluation every time and training on the other 9 folds.

# In[7]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

target = 'count'

# create a feature list for each modeling - experiment by adding features to the exclusion list
features = [feat for feat in list(bike_df_model_ready) if feat not in [target, 'instant', 'dteday']]  

# define cross_validation method (checking the performance of our linear regression model)
cross_validation = KFold(n_splits=10, random_state=42, shuffle=True)

# instantiate LinearRegression() object
reg = linear_model.LinearRegression()

scores = cross_val_score(reg, bike_df_model_ready[features], bike_df_model_ready[target], scoring='neg_mean_squared_error',
                        cv=cross_validation)

rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Std:", scores.std())


# In[8]:


display_scores(rmse_scores)


# The mean of the cross-validated rmse scores shows that our linear regression model is generalizing

# As seen in the text, a way to enhance the performance of our linear regression model is to add powers to the coefficients or existing features. This will add flexibility to our model and will allow us to model non-linear relationships. The disadvantage of polynomial features is that it can lead to a model that overfits the target data (overfitting) and won't generalize well when making predictions on unseen data. A model is overfitting the training data when it fits the training dataset too closely. A closely fit model is one that begins to capture the noise within the dataset. This generally occurs because the can be entirely too complicated or has too many features, that is, the more features that are added the more likely you are to overfit the training dataset. In the case of polynomial features, as the degree increases so does the number of features or terms. Increasing the degree allows the model to have more turning points (d-1, where d is the degree number). 

# ### Linear Regression w/ Polynomial Features

# In[12]:


# make a copy for editing without affecting original
bike_df_model_ready = bikes_hour_df.copy()

target = 'count'
# create a feature list for eash modeling - experiment by adding features to the exclusion list
features = [feat for feat in list(bike_df_model_ready) if feat not in [target, 'instant', 'dteday']]  

# split data into train and test portions and model
X_train, X_test, y_train, y_test = train_test_split(bike_df_model_ready[features], 
                                                 bike_df_model_ready[['count']], 
                                                 test_size=0.2, random_state=42)

# instantiate PolynomialFeatures Class
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test) #not using fit_transform on test because the model won't see this data

from sklearn import linear_model
poly_reg = linear_model.LinearRegression()
 
# train the model on training set
poly_reg.fit(X_train, y_train)

# check model accuracy score
print("Adjusted R^2: %.2f" % (1 - (1-poly_reg.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)))

# make predictions using the testing set
predictions = poly_reg.predict(X_test)
 
# root mean squared error
print("Root Mean squared error with PolynomialFeatures set to 2 degrees: %.2f" % sqrt(mean_squared_error(y_test, predictions)))


# In[17]:


# make a copy for editing without affecting original
bike_df_model_ready = bikes_hour_df.copy()

target = 'count'
# create a feature list for eash modeling - experiment by adding features to the exclusion list
features = [feat for feat in list(bike_df_model_ready) if feat not in [target, 'instant', 'dteday']]  

# split data into train and test portions and model
X_train, X_test, y_train, y_test = train_test_split(bike_df_model_ready[features], 
                                                 bike_df_model_ready[['count']], 
                                                 test_size=0.2, random_state=42)

# Instantiate PolynomialFeatures Class
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(3)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test) #not using fit_transform on test because the model won't see this data
 
from sklearn import linear_model
poly_reg_best = linear_model.LinearRegression()
 
# train the model on training set
poly_reg_best.fit(X_train, y_train)

# check model accuracy score
print("Adjusted R^2: %.2f" % (1 - (1-poly_reg_best.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)))

# make predictions using the testing set
predictions = poly_reg_best.predict(X_test)
 
# root mean squared error
print("Root Mean squared error with PolynomialFeatures set to 3 degrees: %.2f" % sqrt(mean_squared_error(y_test, predictions)))


# In[14]:


# make a copy for editing without affecting original
bike_df_model_ready = bikes_hour_df.copy()

target = 'count'
# create a feature list for eash modeling - experiment by adding features to the exclusion list
features = [feat for feat in list(bike_df_model_ready) if feat not in [target, 'instant', 'dteday']]  

# split data into train and test portions and model
X_train, X_test, y_train, y_test = train_test_split(bike_df_model_ready[features], 
                                                 bike_df_model_ready[['count']], 
                                                 test_size=0.2, random_state=42)

# Instantiate PolynomialFeatures Class
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(4)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test) 
 
from sklearn import linear_model
poly_reg_final = linear_model.LinearRegression()
 
# train the model on training set
poly_reg_final.fit(X_train, y_train)

# check model accuracy score
print("Adjusted R^2: %.2f" % (1 - (1-poly_reg_final.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)))

# make predictions using the testing set
predictions = poly_reg_final.predict(X_test)
 
# root mean squared error
print("Root Mean squared error with PolynomialFeatures set to 4 degrees: %.2f" % sqrt(mean_squared_error(y_test, predictions)))


# In[36]:


plot_learning_curves(poly_reg_best, X_train, y_train)


# In[20]:


poly_reg_best_acc = 108.74
reg_acc = 139.21
delta_val = reg_acc-poly_reg_best_acc
print((delta_val*100)/reg_acc)


# Our poly_reg_best model shows about a 22% increase in model accuracy from our initial reg model.

# ### Random Forest

# In[19]:


# make a copy for editing without affecting original
bike_df_model_ready = bikes_hour_df.copy()

target = 'count'
# create a feature list for eash modeling - experiment by adding features to the exclusion list
features = [feat for feat in list(bike_df_model_ready) if feat not in [target, 'instant', 'dteday']]  

# split data into train and test portions and model
X_train, X_test, y_train, y_test = train_test_split(bike_df_model_ready[features], 
                                                 bike_df_model_ready[['count']], 
                                                 test_size=0.2, random_state=42)


# Instantiate RandomForestRegressor() Class
from sklearn.ensemble import RandomForestRegressor

#n_estimators = 100 default (increasing this hypmter gives better performance but makes code slower)
#criterion = squared_error default
#max_depth = [2,3,4,5,6,7,8,9,10] (if this value is too low the model will underfit and if too high the model will overfit)
#min_samples_split = 2 default
#min_samples_leaf = 2 default
#bootstrap = True default (resampling from dataset)
#warm_start = True (iteratively adds more and more trees to the forest. Keeps all the previous trees and add on more trees.)

forest_regressor = RandomForestRegressor(random_state=42, max_depth=15, warm_start=True)
 
# train the model on training set
forest_regressor.fit(X_train, y_train)

# check model accuracy score
print("Adjusted R^2: %.2f" % (1 - (1-forest_regressor.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)))

# make predictions using the testing set
predictions = forest_regressor.predict(X_test)
 
# root mean squared error
print("Root Mean squared error with RandomForestRegressor: %.2f" % sqrt(mean_squared_error(y_test, predictions)))


# In[ ]:


plot_learning_curves(forest_regressor, X_train, y_train)


# In[6]:


forest_regressor_acc = 42.07
reg_acc = 139.21
delta_val = reg_acc-forest_regressor_acc
print((delta_val*100)/reg_acc)


# Our forest regressor model shows about a 70% increase in model accuracy from our initial reg model with our model 99% of the variation in our outcome variable. This is not a surprise for an ensemble algorithm.

# In[37]:


from sklearn.tree import plot_tree
plot_tree(forest_regressor.estimators_[0], filled=True, impurity=True, rounded=True)


# In[28]:


forest_regressor.get_params()


# Now we will use RandomizedSearchCV on our RandomForestRegressor model.

# ### Random Forest w/ RandomizedSearchCV and Hyperparameter Tuning

# In[7]:


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2500, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 25]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 8, 12]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
forest_regressor = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random_search = RandomizedSearchCV(estimator = forest_regressor, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_random_search.fit(X_train, y_train)

# check model accuracy score
print("Adjusted R^2: %.2f" % (1 - (1-rf_random_search.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)))

# make predictions using the testing set
predictions = rf_random_search.predict(X_test)
 
# root mean squared error
print("Root Mean squared error with RandomForestRegressor using RandomizedSearchCV: %.2f" % sqrt(mean_squared_error(y_test, predictions)))


# In[ ]:


plot_learning_curves(rf_random_search, X_train, y_train)


# In[5]:


rf_random_search_acc = 53.28
reg_acc = 139.21
delta_val = reg_acc-rf_random_search_acc
print((delta_val*100)/reg_acc)


# Our rf_random_search model shows about a 62% increase in model accuracy from our initial reg model with our model explaining 100% of the variation in our outcome variable. Getting a model that fits our data perfectly with the randomsearchCV technique is not surprising. This is good but it does not give any information about how our model will perform on unseen data. In this project our concern is prediction power. 

# ## Modeling With Feature Engineering (Creating Dummy Features From Category Variables)

# RandomForestRegressor model performs a lot better than our previous models. We possibly can achieve an even better model by encoding the weathersit categorical feature by giving it binary data points that would make sense to our model,

# ### Linear Regression w/ Encoded Categorical Variables

# In[7]:


def prepare_data_for_model(raw_dataframe, 
                           target_columns, 
                           drop_first = False, 
                           make_na_col = True):
    
    # dummy all categorical fields 
    dataframe_dummy = pd.get_dummies(raw_dataframe, columns=target_columns, 
                                     drop_first=drop_first, 
                                     dummy_na=make_na_col)
    return (dataframe_dummy)

# make a copy for editing without affecting original
bike_df_model_ready = bikes_hour_df.copy()
bike_df_model_ready = bike_df_model_ready.sort_values('instant')

# dummify categorical columns
bike_df_model_ready = prepare_data_for_model(bike_df_model_ready, 
                                            target_columns = ['season', 
                                                              'weekday', 
                                                              'weathersit'],
                                            drop_first = True)

# remove the nan colums in dataframe as most are outcome variable and we can't use them
bike_df_model_ready = bike_df_model_ready.dropna() 

target = 'count'
features = [feat for feat in list(bike_df_model_ready) if feat not in [target, 'instant',  'dteday']]  

 
X_train, X_test, y_train, y_test = train_test_split(bike_df_model_ready[features], 
                                                 bike_df_model_ready[['count']], 
                                                 test_size=0.2, 
                                                 random_state=42)
from sklearn import linear_model
reg_enc = linear_model.LinearRegression()
 
# train the model on training set
reg_enc.fit(X_train, y_train)

# check model accuracy score
print("Adjusted R^2: %.2f" % (1 - (1-reg_enc.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)))

# make predictions using the testing set
predictions = reg_enc.predict(X_test)

# print coefficients as this is what our web application will use in the end
print('Coefficients: \n', reg_enc.coef_)

# root mean squared error
print("Root Mean squared error: %.2f" % sqrt(mean_squared_error(y_test, predictions)))


# This proves that encoding categorical features positively impacts our model's ability to predict with greater accuracy hence the increase in our rmse score with our original linear regression model.

# In[25]:


bike_df_model_ready[['weathersit_2.0', 'weathersit_3.0', 'weathersit_4.0']].head()


# In[ ]:


plot_learning_curves(reg_enc, X_train, y_train)


# ### Gradient Boosting w/ Encoded Categorical Variables
# 
# The alternative to bagging (bootstrap aggregation) or chosen samples with replacement, combining them, and then taking their average, is Boosting. Instead of aggregating predictions, boosters will turn weak learners into strong learnings by focusing on where the individual models went wrong. In Gradient Boosting, individual models train upon the residuals, which is the point of error between the prediction and the actual results. Instead of aggregating trees, gradient boosted trees learns from these point of errors during each boosting round. Essentially, bagging is a method of merging the same type of predictions. Boosting is a method of merging different types of predictions. Bagging decreases variance, not bias, and solves over-fitting issues in a model. Boosting decreases bias, not variance. 

# In[38]:


get_ipython().system('pip install scikit-optimize')


# In[51]:


# simple approach - make a copy for editing without affecting original
bike_df_model_ready = bikes_hour_df.copy()
bike_df_model_ready = bike_df_model_ready.sort_values('instant')

# dummify categorical columns
bike_df_model_ready = prepare_data_for_model(bike_df_model_ready, 
                                             target_columns = ['season', 'weekday', 'weathersit'])
list(bike_df_model_ready.head(1).values)

# remove the nan colums in dataframe as most are outcome variable and we can't use them
bike_df_model_ready = bike_df_model_ready.dropna() 


target = 'count'
features = [feat for feat in list(bike_df_model_ready) if feat not in [target, 'instant', 'dteday']]  

 
X_train, X_test, y_train, y_test = train_test_split(bike_df_model_ready[features], 
                                                 bike_df_model_ready[['count']], 
                                                 test_size=0.2, 
                                                 random_state=42)
 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
# from skopt import BayesSearchCV
# skopt output is raising an exception error so i will result back to randomsearchCV


param_grid = {'max_depth': [2,4,6,8,10,15,20],
              'n_estimators': [5,10,15,20],
              'learning_rate': [0.2,0.4,0.6,0.8,0.95,1.0]}


model_gbr = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), param_grid, n_iter=50, cv=10, verbose= 3) 

import timeit
print(model_gbr.fit(X_train, np.ravel(y_train)), timeit.timeit())

# check model accuracy score
print("Adjusted R^2: %.2f" % (1 - (1-model_gbr.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)))

predictions = model_gbr.predict(X_test)

# root mean squared error
print("Root Mean squared error: %.2f" % sqrt(mean_squared_error(y_test, predictions)))


# In[52]:


model_gbr.best_params_


# In[53]:


model_gbr = GradientBoostingRegressor(random_state=42, n_estimators=20, max_depth=10, learning_rate=0.2)

print(model_gbr.fit(X_train, np.ravel(y_train)))

# check model accuracy score
print("Adjusted R^2: %.2f" % (1 - (1-model_gbr.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)))

predictions = model_gbr.predict(X_test)

# root mean squared error
print("Root Mean squared error: %.2f" % sqrt(mean_squared_error(y_test, predictions)))


# In[ ]:


plot_learning_curves(model_gbr, X_train, y_train)


# In[ ]:


model_gbr = 41.18
reg_acc = 139.21
delta_val = reg_acc-model_gbr
print((delta_val*100)/reg_acc)


# ### XGBoost w/ Encoded Categorical Variables
# 
# eXtreme Gradient Boosting is an exaggerated version of simple Gradient Boosting in that XGBoost is equipped with speed enhancements such as parallel computing and cache awareness that makes the algorithm approximately 10 times faster than traditional GB. On average, Boosting performs better than Bagging.

# In[28]:


get_ipython().system('pip install xgboost')


# In[9]:


from xgboost import XGBRegressor

# simple approach - make a copy for editing without affecting original
bike_df_model_ready = bikes_hour_df.copy()
bike_df_model_ready = bike_df_model_ready.sort_values('instant')

# dummify categorical columns
bike_df_model_ready = prepare_data_for_model(bike_df_model_ready, 
                                             target_columns = ['season', 'weekday', 'weathersit'])
list(bike_df_model_ready.head(1).values)

# remove the nan colums in dataframe as most are outcome variable and we can't use them
bike_df_model_ready = bike_df_model_ready.dropna() 


target = 'count'
features = [feat for feat in list(bike_df_model_ready) if feat not in [target, 'instant', 'dteday']]  

 
X_train, X_test, y_train, y_test = train_test_split(bike_df_model_ready[features], 
                                                 bike_df_model_ready[['count']], 
                                                 test_size=0.5, 
                                                 random_state=42)
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, np.ravel(y_train))

# check model accuracy score
print("Adjusted R^2: %.2f" % (1 - (1-xgb_model.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)))

predictions = xgb_model.predict(X_test)

# root mean squared error
print("Root Mean squared error: %.2f" % sqrt(mean_squared_error(y_test, predictions)))


# In[ ]:


plot_learning_curves(xgb_model, X_train, y_train)


# Just using the default hyperparameters and no cross validation optimization technique, our xgboost model rmse score is better than our randomsearchCV (random forest regressor estimator) model rmse score. However, our random forest regressor model with just default hyperparameter values has a slightly better rmse score than our xgboost model and a higher adjusted r squared value by 0.01. 

# ### KNeighborsRegressor w/ Encoded Categorical Variables
# 
# Our dataset is farely small so using KNN is attractive. According to Prakash Nadkarni in Clinical Research Computing, KNN is a standard machine learning method with the idea that one uses a large amount of training data, where each data point is characterized by a set of variables. Essentially, each point is plotted in a high-dimensional space, where each axis in the space corresponds to an individual variable. So, when there is a new test data point, we want to find out the K nearest neighbors that are closest or most similar to it. It is stated that the number K is usually chosen as the square root of N or the total number of observations in the training data set. 

# In[10]:


from sklearn.neighbors import KNeighborsRegressor

# simple approach - make a copy for editing without affecting original
bike_df_model_ready = bikes_hour_df.copy()
bike_df_model_ready = bike_df_model_ready.sort_values('instant')

# dummify categorical columns
bike_df_model_ready = prepare_data_for_model(bike_df_model_ready, 
                                             target_columns = ['season', 'weekday', 'weathersit'])
list(bike_df_model_ready.head(1).values)

# remove the nan colums in dataframe as most are outcome variable and we can't use them
bike_df_model_ready = bike_df_model_ready.dropna() 


target = 'count'
features = [feat for feat in list(bike_df_model_ready) if feat not in [target, 'instant', 'dteday']]  

 
X_train, X_test, y_train, y_test = train_test_split(bike_df_model_ready[features], 
                                                 bike_df_model_ready[['count']], 
                                                 test_size=0.5, 
                                                 random_state=42)

# standardizing features because kNN uses euclidean distance to measure the distance between points
# which is the square root of the sum of the squared differences between two points. Not standardizing would force the 
# calculation to be heavily weighted on the difference in ratios of the features. Even if the scale of features is not different
# it is still good practice to standardize features when using this algorithm

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

knn = KNeighborsRegressor(n_neighbors= int(np.sqrt(len(X_train))))
knn.fit(X_train, y_train)

# check model accuracy score
print("Adjusted R^2: %.2f" % (1 - (1-knn.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)))

predictions = knn.predict(X_test)

# root mean squared error
print("Root Mean squared error: %.2f" % sqrt(mean_squared_error(y_test, predictions)))


# In[11]:


k_values = [n for n in range(1,200)]
cv_scores = []

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)

from sklearn.model_selection import cross_val_score
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    score = cross_val_score(knn, X_train, y_train, cv=10)
    cv_scores.append(np.mean(score))

plt.plot(k_values, cv_scores, marker='o')
plt.xlabel("k_values")
plt.ylabel("Accuracy Score")


# In[54]:


#training model using the best k that cross_val_score gave us
best_index_value = np.argmax(cv_scores)
best_k = k_values[best_index_value]
knn = KNeighborsRegressor(n_neighbors=best_k)
knn.fit(X_train, y_train)

# check model accuracy score
print("Adjusted R^2: %.2f" % (1 - (1-knn.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)))

predictions = knn.predict(X_test)

# root mean squared error
print("Root Mean squared error: %.2f" % sqrt(mean_squared_error(y_test, predictions)))


# In[ ]:


plot_learning_curves(knn, X_train, y_train)


# ## Implementing Time-Series Feature Engineering Technique
# 
# According to the text, since the data is a ledger of bike rentals over time, it technically is a time-series dataset. The textbook states that whenever the dataset records events over time, recognize it as an added feature. It is stated that time captures trends, changing needs and perceptions, etc. The idea here is that we want to create features that capture all of these time-evolving elements. So, as stated in the text, for each row of data, we'll add two new features: the sum of bicycle rentals for the previous hour, and the sum of bicycle rentals from two hours ago. The idea is that if we want to understand the current bicycling mood, we can start by looking at what happened an hour ago.

# In[40]:


# prior hours
bikes_hour_df_shift = bikes_hour_df[['dteday','hr','count']].groupby(['dteday','hr']).sum().reset_index()
bikes_hour_df_shift.sort_values(['dteday','hr'])

# shift the count of the last two hours forward so the new count can take in consideratio how the last two hours went 
bikes_hour_df_shift['sum_hr_shift_1'] = bikes_hour_df_shift[['count']].shift(+1)
bikes_hour_df_shift['sum_hr_shift_2'] = bikes_hour_df_shift[['count']].shift(+2)

bike_df_model_ready =  pd.merge(bikes_hour_df, bikes_hour_df_shift[['dteday', 'hr', 'sum_hr_shift_1', 'sum_hr_shift_2']], how='inner', on = ['dteday', 'hr'])

# drop NAs caused by our shifting fields around
bike_df_model_ready = bike_df_model_ready.dropna()

target = 'count'
# create a feature list for each modeling - experiment by adding features to the exclusion list
features = [feat for feat in list(bike_df_model_ready) if feat not in [target, 'instant', 'dteday','casual', 'registered']]  

# split data into train and test portions and model
X_train, X_test, y_train, y_test = train_test_split(bike_df_model_ready[features], 
                                                 bike_df_model_ready[['count']], 
                                                 test_size=0.3, random_state=42)

from sklearn.ensemble import GradientBoostingRegressor
model_gbr_final = GradientBoostingRegressor(learning_rate=0.2, max_depth=10, n_estimators=20,
                          random_state=42)
model_gbr_final.fit(X_train, np.ravel(y_train))

# check model accuracy score
print("Adjusted R^2: %.2f" % (1 - (1-model_gbr_final.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)))

predictions = model_gbr_final.predict(X_test)

# root mean squared error
print("Root Mean squared error: %.2f" % sqrt(mean_squared_error(y_test, predictions)))


# In[ ]:


plot_learning_curves(model_gbr, X_train, y_train)


# In[42]:


model_gbr_final_acc = 33.26
reg_acc = 139.21
delta_val = reg_acc-model_gbr_final_acc
print((delta_val*100)/reg_acc)


# GradientBoostingRegressor with the added time shift features is by far our best model with just about 100% of the data points being within the regression line and a 76% increase in our prediction accuracy. Checking for outliers and variable skewness would also be a valid technique that may further increase this model's prediction accuracy.

# # Creating A Suitable Model For Our Web Application

# In[59]:


# loop through each feature and calculate the R^2 score
features = ['hr', 'season', 'holiday', 'temp']
from sklearn import linear_model
from sklearn.metrics import r2_score

# split data into train and test portions and model
X_train, X_test, y_train, y_test = train_test_split(bike_df_model_ready[features], 
                                                 bike_df_model_ready[['count']], 
                                                 test_size=0.2, random_state=42)
    
for feat in features:
    model_lr = linear_model.LinearRegression()
    model_lr.fit(X_train[[feat]], y_train)
    predictions = model_lr.predict(X_test[[feat]])
    print('R^2 for %s is %f' % (feat, r2_score(y_test, predictions)))


# In[61]:


# simple approach - make a copy for editing without affecting original
bike_df_model_ready = bikes_hour_df[['hr', 'season', 'holiday', 'temp', 'count']].copy()

outcome = 'count'

# dummify categorical columns
bike_df_model_ready = prepare_data_for_model(bike_df_model_ready,  drop_first = False, 
                                             make_na_col = False, target_columns = ['season'])

features = [feat for feat in bike_df_model_ready if feat not in ['count']]  

# split data into train and test portions and model
X_train, X_test, y_train, y_test = train_test_split(bike_df_model_ready[features], 
                                                 bike_df_model_ready[['count']], 
                                                 test_size=0.2, random_state=42)
from sklearn import linear_model
model_lr = linear_model.LinearRegression()

for feat in features:
    model_lr = linear_model.LinearRegression()
    # train the model on training set
    model_lr.fit(X_train[[feat]], y_train)
    predictions = model_lr.predict(X_test[[feat]])
    
    print('R^2 for %s is %f' % (feat, r2_score(y_test, predictions)))


# In[62]:


# train the model on training set
model_lr.fit(X_train, y_train)

# make predictions using the testing set
predictions = model_lr.predict(X_test)
 
# root mean squared error
print("Root Mean squared error: %.2f" % sqrt(mean_squared_error(y_test, predictions)))
print('\n')
print('Intercept: %f' % model_lr.intercept_)

# features with coefficients 
feature_coefficients  = pd.DataFrame({'coefficients':model_lr.coef_[0], 
                                    'features':X_train.columns.values})

feature_coefficients.sort_values('coefficients')


# In[43]:


# set up constants for our coefficients 
INTERCEPT = -121.029547
COEF_HOLIDAY = -23.426176   # day is holiday or not
COEF_HOUR = 8.631624        # hour (0 to 23)
COEF_SEASON_1 = 3.861149    # 1:springer
COEF_SEASON_2 = -1.624812   # 2:summer
COEF_SEASON_3 = -41.245562  # 3:fall
COEF_SEASON_4 = 39.009224   # 4:winter
COEF_TEMP = 426.900259      # norm temp in Celsius -8 to +39


# # Making Prediction With Regression Normal Equation (This is the best model that will translate to a user-friendly web app)

# In[44]:


# mean values
MEAN_HOLIDAY = 0.0275   # day is holiday or not
MEAN_HOUR = 11.6        # hour (0 to 23)
MEAN_SEASON_1 = 1       # 1:spring
MEAN_SEASON_2 = 0       # 2:summer
MEAN_SEASON_3 = 0       # 3:fall
MEAN_SEASON_4 = 0       # 4:winter
MEAN_TEMP = 0.4967      # norm temp in Celsius -8 to +39


# try predicting something - 9AM with all other features held constant
rental_counts = INTERCEPT + (MEAN_HOLIDAY * COEF_HOLIDAY) \
    + (9 * COEF_HOUR) \
    + (MEAN_SEASON_1 * COEF_SEASON_1)  + (MEAN_SEASON_2 * COEF_SEASON_2) \
    + (MEAN_SEASON_3 * COEF_SEASON_3)  + (MEAN_SEASON_4 * COEF_SEASON_4) \
    + (MEAN_TEMP * COEF_TEMP)

print('Estimated bike rental count for selected parameters: %i' % int(rental_counts))   


# # Export Final Model
# 
# The GradientBoostingRegressor with the timeshift features is our best model. Xgboost maybe could have been the better model if more time was spent on hyperparameter tuning like on the GradientBoostingRegressor model. Below, we will package up our best model in a pickle file and test it out.

# In[22]:


import joblib
joblib.dump(model_gbr_final, 'model_gbr_final.pkl', compress = 1)


# # Load Model and Predict

# In[27]:


# split data into train and test portions and model (I'm resplitting the data since this particular model has already seen
#X_test)
X_train, X_test, y_train, y_test = train_test_split(bike_df_model_ready[features], 
                                                 bike_df_model_ready[['count']], 
                                                 test_size=0.45, random_state=42)

# create data frame
unseen_data = X_test

# open file
file = open("model_gbr_final.pkl", "rb")

# load trained model
trained_model = joblib.load(file)

# predict
prediction = trained_model.predict(unseen_data)


# In[34]:


pd.DataFrame({'original bike count': np.ravel(y_test), 'predicted bike count': prediction})


# Of course this model won't translate well into a web app with a user-interface. It makes sense to allow the user to pick a feature data point from one index but the model won't learn anything from a single row of data points. That also means that although this model is our best model at predicting future bike count based on the bike count in the past, it is not a very intuitive model. Obviously, the best model that would translate to a user-friendly web app is the linear regression model where the user can input a few coefficient values and expect a single bike prediction based on their input.

# In[ ]:




