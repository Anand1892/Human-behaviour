#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #Data processing and I/O operation
import numpy as np #Linear Algebra
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

#Import the machine libraries
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[2]:


train = shuffle(pd.read_csv('C://Users//deep//Desktop//Decodr//Case Studies_ Practice Files_ Reference Materials//Case Studies//Additional Solved Projects//Human behaviour project//train.csv'))
test = shuffle(pd.read_csv('C://Users//deep//Desktop//Decodr//Case Studies_ Practice Files_ Reference Materials//Case Studies//Additional Solved Projects//Human behaviour project//test.csv'))


# In[3]:


train.head(10)


# In[5]:


train.tail(10)


# In[6]:


train.shape


# In[8]:


train.isnull().values.any()


# In[9]:


test.isnull().values.any()


# In[10]:


train_outcome = pd.crosstab(index = train['Activity'], columns='Count')
train_outcome


# In[11]:


temp = train['Activity'].value_counts()
temp


# In[12]:


df = pd.DataFrame({'labels':temp.index, 'values':temp.values})


# In[13]:


df.head(2)


# In[14]:


labels=df['labels']
sizes = df['values']
colors = ['yellowgreen', 'lightskyblue', 'gold', 'lightpink', 'cyan', 'lightcoral']
patches, texts = plt.pie(sizes, colors=colors, labels = labels, shadow=True, startangle=90, pctdistance=1.1, labeldistance=1.2)
plt.legend(patches, labels, loc='right')
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[15]:


X_train = pd.DataFrame(train.drop(['Activity', 'subject'], axis=1))
Y_train_label = train.Activity.values.astype(object)
X_test = pd.DataFrame(test.drop(['Activity', 'subject'], axis=1))
Y_test_label = test.Activity.values.astype(object)


# In[16]:


from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()

encoder.fit(Y_train_label)
y_train = encoder.transform(Y_train_label)


# In[17]:


y_train


# In[18]:


encoder.fit(Y_test_label)
y_test = encoder.transform(Y_test_label)


# In[19]:


y_test


# In[20]:


num_cols = X_train._get_numeric_data().columns
num_cols.size


# In[21]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_test = scaler.fit_transform(X_test)


# In[22]:


knn = KNeighborsClassifier(n_neighbors=24)
knn.fit(x_train,y_train)


# In[23]:


y_pred = knn.predict(x_test)


# In[24]:


print((accuracy_score(y_test, y_pred)*100), '%')


# In[ ]:


scores = []
for i in range(1,50):
    knn=KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores.append(accuracy_score(y_test, y_pred))


# In[ ]:


plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy Score')
xticks = range(1,50)
plt.plot(xticks, scores, color='red', linestyle='solid', marker='o', 
        markersize=5, markerfacecolor='blue')
plt.show()


# In[ ]:


scores = np.array(scores)
print('Optimal number of neighbors is:', scores.argmax())
print('Accuracy Score:' +str(scores.max()*100),'%')


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
y_pred_label = list(encoder.inverse_transform(y_pred))


# In[ ]:


print(confusion_matrix(Y_test_label, y_pred_label))


# In[ ]:


print(classification_report(Y_test_label, y_pred_label))


# In[ ]:




