#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import pickle


df=pd.read_csv("IPL.csv")


# In[2]:


df


# In[3]:


df.head(10)


# In[4]:


df.shape


# In[5]:


df.size


# In[6]:


df.columns


# In[7]:


df.dtypes


# In[8]:


pd.isnull(df)


# In[9]:


df.isna().sum()


# In[10]:


#data cleaning 
columns_to_remove=['mid','venue', 'batsman','bowler','striker','non-striker']
df.drop(labels=columns_to_remove,axis=1,inplace=True)


# In[11]:


df.head()


# In[12]:


df['bat_team'].unique()


# In[13]:


consistent_team=['Kolkata Knight Riders','Chennai Super Kings','Rajasthan Royals',
                 'Mumbai Indians','Kings XI Punjab','Royal Challengers Banglore','Sunrisers Hyderabad'
                 ,'Delhi Daredevils']


# In[14]:


df=df[(df['bat_team'].isin(consistent_team)) & (df['bowl_team'].isin(consistent_team))]


# In[15]:


#removing the first 5 over from each row
df=df[df['overs']>=5.0]


# In[16]:


df.head()


# In[17]:


print(df['bat_team'].unique())
print(df['bowl_team'].unique())


# In[18]:


from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%d-%m-%Y'))


# In[19]:


encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])


# In[20]:


encoded_df


# In[21]:


encoded_df.columns


# In[24]:


encoded_df=encoded_df[['date', 'bat_team_Chennai Super Kings',
       'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
       'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians',
       'bat_team_Rajasthan Royals', 'bat_team_Sunrisers Hyderabad',
       'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils',
       'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
       'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
       'bowl_team_Sunrisers Hyderabad', 'runs', 'wickets', 'overs', 'runs_last_5',
       'wickets_last_5', 'total']]


# In[25]:


#splitting the data
X_train=encoded_df.drop(labels='total',axis=1)[encoded_df['date'].dt.year<=2016]
X_test=encoded_df.drop(labels='total',axis=1)[encoded_df['date'].dt.year>=2017]


# In[26]:


Y_train=encoded_df[encoded_df['date'].dt.year <=2016]['total'].values
Y_test=encoded_df[encoded_df['date'].dt.year >=2017]['total'].values


# In[27]:


X_train.drop(labels='date',axis=1,inplace=True)
X_test.drop(labels='date',axis=1,inplace=True)


# In[28]:


#linear regression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_test,Y_test)


# In[ ]:





# In[ ]:





# In[31]:


##Ridge regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[32]:


ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,Y_train)


# In[33]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[34]:


prediction=ridge_regressor.predict(X_test)


# In[35]:


import seaborn as sns
sns.distplot(Y_test-prediction)


# In[41]:


from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(Y_test,prediction))
print('MSE:', metrics.mean_squared_error(Y_test,prediction))
print('RMSE:',np.sqrt(metrics.mean_squared_error(Y_test,prediction)))


# In[43]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# In[44]:


lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(X_train,Y_train)
print(lasso_regressor.best_params_)


# In[45]:


print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[46]:


pred=lasso_regressor.predict(X_test)


# In[48]:


import seaborn as sns
sns.distplot(Y_test-pred)


# In[49]:


print('MAE:',metrics.median_absolute_error(Y_test,pred))
print('MSE:',metrics.mean_squared_error(Y_test,pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(Y_test,pred)))


# In[50]:


# Creating a pickle file for the classifier
filename = 'first-innings-score-lr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))


# In[ ]:




