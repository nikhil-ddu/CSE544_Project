#!/usr/bin/env python
# coding: utf-8

# #### Hypothesis: amount of loan lent per month is a time series. We try to predict total amount that will be lent on a month

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[90]:


import matplotlib.pyplot as plt
import pylab


# In[67]:


df = pd.read_csv("Master_Loan_Summary.csv")


# In[68]:


df = df[["origination_date","amount_borrowed","data_source"]]


# In[69]:


df.head()


# In[70]:


def make_feature(df):
    
    date = [] 
    #months = []
    
    for index, row in df.iterrows():
                
        #Calculate year and month
        year,month,day = row['origination_date'].split('-')     
        
        date.append(year+month)
        #months.append(month)
                 
        
    df["date"] = date
    #df["month"] = months
    
    return df


# In[71]:


df = make_feature(df)


# In[72]:


df.head()


# In[73]:


print(df["date"].unique())


# In[74]:


null_count = df.isna().sum()
null_count


# In[75]:


grouped = df.groupby(['date'])['amount_borrowed'].sum().reset_index()


# In[78]:


grouped.head()


# In[122]:


x = grouped.values


# In[123]:


y = x[:,0]


# In[124]:


x = x[:,1]


# In[134]:


def plot_data(r,s,y_pred,test,y_label):
    
    plot_x = []
    for i in range(r,r+s):
        plot_x.append(i)
        
    plt.plot(plot_x,test,label="original")
    plt.plot(plot_x,y_pred,label=y_label)
    plt.xlabel('Time')
    plt.ylabel('Loan Borrowed per Month')
    pylab.legend(loc='upper left')   
    plt.show()


# In[137]:


def EWMA(alpha):
    
    training = x[:25]
    test = x[25:]
    
    r= len(training)
    s = len(test)
    
    error = []
    y = []
    for i in range(r,r+s):
        base = 0
        j = i-1
        k = 0
        while(j >= 0):
            base = base+ (x[j]*((1-alpha)**k))
            j = j-1 
            k = k+1

        y_pred = alpha*base
        y.append(y_pred)
        
        error.append((abs(y_pred-x[i])/x[i])*100)
        
    plot_data(r,s,y,test,"EWMA("+str(alpha)+")")
    
    return sum(error)/len(error)


# In[138]:


error = EWMA(0.5)
print(error)


# In[111]:


def train_lr(p,length):
    #print("jhjd")
    training = []
    label = []
    
    for i in range(length):
        if(i+p < length):
            #print("ghttt")
            training.append([1])
            training[i] = training[i]+list(x[i:i+p])
            #training.append(x[i:i+144])
            label.append(x[i+p])

        else:
            break
            
    #print(len(label))
    beta=np.matmul(np.linalg.inv(np.matmul(np.transpose(training),training)),np.matmul(np.transpose(training),label))
    
    return beta


# In[112]:


def AR(p):
    
    training = x[:25]
    test = x[25:]
    
    r= len(training)
    s = len(test)
    
    error = []
    y = []
    for i in range(r,r+s):
        
        #print(i)
        testx = [1]
        temp = x[i-p:i]
        testx = testx + list(temp)
        #print("ggg")
        beta = train_lr(p,i)
        y_pred=np.matmul(testx,beta)
        y.append(y_pred)
        
        error.append((abs(y_pred-x[i])/x[i])*100)
        
    plot_data(r,s,y,test,"AR("+str(p)+")")
    
    return sum(error)/len(error)


# In[115]:


AR(2)


# In[116]:


def Seasonal(S):
    
    training = x[:25]
    test = x[25:]
    
    r= len(training)
    s = len(test)
    
    error = []
    y = []
    for i in range(r,r+s):
      
        y_pred = x[i-S]
        y.append(y_pred)

        error.append((abs(y_pred-x[i])/x[i])*100)
        
    plot_data(r,s,y,test,"Seasonal("+str(S)+")")
    
    return sum(error)/len(error)


# In[121]:


Seasonal(3)

