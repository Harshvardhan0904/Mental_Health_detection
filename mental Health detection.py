#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("C:/Users/harsh/Downloads/Combined Data.csv")
df.head(5)


# In[3]:


df.drop(['Unnamed: 0'], axis = 1 , inplace = True)


# In[4]:


df.head()


# In[5]:


status = df['status'].unique()


# In[6]:


condition = []
for i in status :
    condition.append(i)
condition


# In[7]:


import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[8]:


stop_words = set(stopwords.words("english"))


# In[10]:


def text_clean(text):
    # Lowercase the text
    text = str(text).lower()
    return text 
df['statement']=df['statement'].apply(text_clean)


# In[11]:


df['statement'].shape


# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# In[13]:


x = df['statement']
y = df['status']


# In[14]:


le = LabelEncoder()
y = le.fit_transform(y)
x_train , x_test , y_train ,y_test = train_test_split(x,y , stratify=y  ,test_size= 0.2 , random_state=42)
print(len(x_test) , len(y_test))


# In[41]:


tf = TfidfVectorizer(max_features=5000 ,max_df= 0.85 , min_df=1 , stop_words='english')
x_train_vect = tf.fit_transform(x_train)
x_test_vect = tf.transform(x_test)


# In[42]:


print(x_test_vect.shape , len(y_test))


# In[51]:


from sklearn.ensemble import RandomForestClassifier

# Example of setting parameters
rf = RandomForestClassifier()

# Fit the model on training data
rf.fit(x_train_vect, y_train)


# In[52]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[53]:


pred = rf.predict(x_test_vect)


# In[54]:


acc = accuracy_score(y_test , pred)
acc


# In[55]:


status_list  = {label : index for index , label in enumerate(le.classes_)}


# In[56]:


status_list


# In[77]:


def pred_probel(text , vector , model):
    if len(text)==0 :
        print("Cant predict since no text provided ")
        return None
    text = text_clean(text)
    statement_vector  = tf.transform([text])
    pred = rf.predict(statement_vector)
    return pred[0]


# In[78]:


text = ' '
res = pred_probel(text,x_train_vect,rf)
print(res)


# In[39]:





# In[40]:


sen


# In[ ]:




