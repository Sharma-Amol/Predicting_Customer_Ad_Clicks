#!/usr/bin/env python
# coding: utf-8

# # Context
# 
# In this project I will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an advertisement. I'll create a model that will predict whether or not users will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad
# 
# ## Dataset
# 
# [advertising.csv](https://github.com/Sharma-Amol/Predicting_Customer_Ad_Clicks/blob/main/advertising.csv)

# # Let's get started.

# ## Importing relevant Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Geting the Data

# In[2]:


ad_data = pd.read_csv('advertising.csv')


# In[3]:


# Checking head of dataset.

ad_data.head()


# In[4]:


# To know data types of column values.

ad_data.info()


# In[5]:


# Quick statistical report.

ad_data.describe()


# In[6]:


# To know column labels.

ad_data.columns


# In[7]:


# Row labels.

ad_data.index


# ## Exploratory Data Analysis
# 
# Performing EDA using Seaborn for visulaization.

# In[8]:


# Creating a histogram of the Age

plt.figure(figsize=(10,10),dpi=100)
sns.set_style('whitegrid')
sns.displot(ad_data['Age'],bins=30)


# In[9]:


# Creating a jointplot showing Area Income versus Age.

plt.figure(figsize=(10,10),dpi=100)
sns.jointplot(x=ad_data['Age'],y=ad_data['Area Income'])


# There is a trend here. One can start earning when he/she get in his/her 20s. As you grow older, your income starts to increase and then, towards retirement, you get no income or starts to drop. Can explore this a little more.

# In[10]:


# Creating a jointplot showing the kde distributions of Daily Time spent on site vs. Age.

plt.figure(figsize=(10,10),dpi=100)
sns.jointplot(x=ad_data['Age'],y=ad_data['Daily Time Spent on Site'],kind='kde',shade = True)


# In[11]:


# Creating a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

plt.figure(figsize=(10,10),dpi=100)
sns.jointplot(x=ad_data['Daily Time Spent on Site'],y=ad_data['Daily Internet Usage'])


# In[12]:


# Finally, creating a pairplot with the hue defined by the 'Clicked on Ad' column feature.

plt.figure(figsize=(10,10),dpi=100)
sns.pairplot(data=ad_data,hue='Clicked on Ad')


# In[13]:


plt.figure(figsize=(10,10),dpi=100)
sns.pairplot(data=ad_data,hue='Clicked on Ad',diag_kind='hist')
plt.savefig("Ad Clicks Pairplot.jpg")


# # Logistic Regression
# 
# Performing train test split on dataset, and training the model.

# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[17]:


from sklearn.linear_model import LogisticRegression


# In[18]:


logdata = LogisticRegression()


# ## Predictions and Evaluations

# In[19]:


logdata.fit(X_train,y_train)
predictions = logdata.predict(X_test)


# # Creating a classification report for the model.

# In[20]:


from sklearn.metrics import classification_report,confusion_matrix


# In[21]:


print (classification_report(y_test,predictions))
print (confusion_matrix(y_test,predictions))


# # Conclusion
# 
# sklearn's confusion matrix :-
# ![Sklearn's%20confusion%20matrix.jpg](attachment:Sklearn's%20confusion%20matrix.jpg)
# 
# a.) Model is good fit. It has >90% on precision recall and accuracy.
# 
# b.) We have some mislabelled points:-
# 
#     8 were predicted as positive (1) when they were actually negative (0). 
#     
#     14 were predicted as negative (0) when they were actually positive (1).
#     
#     Given the size of dataset, these are acceptable values.
