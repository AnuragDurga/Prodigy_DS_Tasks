#!/usr/bin/env python
# coding: utf-8

# # Perform data cleaning and exploratory data analysis (EDA) on the Titanic dataset from Kaggle. Exploring the relationships between variables and identify patterns and trends in the data.

# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[50]:


data= pd.read_csv('C:/Users/HP/Desktop/Titanic.csv')


# In[51]:


data.info()


# In[52]:


data.describe()


# In[53]:


#Handling missing values
data.isnull().sum()


# In[54]:


data['Age'].fillna(data['Age'].median(), inplace=True)


# In[55]:


data['Cabin'].fillna(data['Cabin'].mode()[0], inplace=True)


# In[56]:


data['Fare'].fillna(data['Fare'].mode()[0], inplace=True)


# In[57]:


data.isnull().sum()


# In[58]:


import seaborn as sns


# In[59]:


#Create a histogram of the 'Age' variable
plt.hist(data['Age'], bins=10)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()


# # Perfroming Exploratory Data Analysis

# In[60]:


plt.figure(figsize=(8,6))
sns.countplot(x='Survived', data=data)
plt.title('Distribution of Survival')
plt.show()


# In[61]:


counts = data.groupby(['Pclass', 'Survived']).size().unstack()
counts.plot(kind="bar")
plt.show()


# In[62]:


print(data.describe())

