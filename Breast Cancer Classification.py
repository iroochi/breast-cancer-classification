#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split


# In[2]:


breast_cancer_df = pd.read_csv('breastcancerdata.csv')


# In[3]:


breast_cancer_df.head()


# In[4]:


breast_cancer_df.columns


# In[5]:


remove_columns = breast_cancer_df[['id', 'diagnosis']]


# In[6]:


remove_columns.head()


# In[7]:


breast_cancer_df = breast_cancer_df.drop(columns = ['id', 'diagnosis'])


# In[8]:


breast_cancer_df.head()


# In[9]:


breast_cancer_df['target'] = remove_columns['diagnosis']


# In[10]:


breast_cancer_df.head()


# In[11]:


breast_cancer_df['target'].unique()


# In[12]:


from sklearn.preprocessing import LabelEncoder


# In[13]:


label_encoder = LabelEncoder()
breast_cancer_df['target'] = label_encoder.fit_transform(breast_cancer_df['target'])


# In[14]:


breast_cancer_df.head()


# 1 - Malignant
# 
# 0 - Benign

# In[15]:


breast_cancer_df.shape


# In[16]:


breast_cancer_df.isnull().sum()


# In[17]:


breast_cancer_df = breast_cancer_df.dropna(axis = 1)


# In[18]:


breast_cancer_df.head()


# In[19]:


breast_cancer_df.describe()


# In[20]:


breast_cancer_df['target'].value_counts()


# In[21]:


breast_cancer_df.groupby('target').mean()


# In[22]:


# separating the features and target
X = breast_cancer_df.drop(columns='target', axis = 1)
Y = breast_cancer_df['target']


# In[23]:


print(X)


# In[24]:


print(Y)


# In[25]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)


# In[26]:


print(X.shape, X_train.shape, X_test.shape)


# In[27]:


# standardize the data
from sklearn.preprocessing import StandardScaler


# In[28]:


scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


# In[29]:


print(X_train_std)


# In[ ]:





# In[30]:


# importing tensorflow and keras
import tensorflow as tf

# Set the seed
tf.random.set_seed(3)
from tensorflow import keras


# In[31]:


# setting up the layers of NN
model = keras.Sequential([
                          keras.layers.Flatten(input_shape = (30,)), # input layer - convert the data into single dimensional array
                          keras.layers.Dense(20, activation='relu'), # hidden layer
                          keras.layers.Dense(30, activation='sigmoid'),
                          keras.layers.Dense(2, activation='sigmoid') # output layer 
                          # the number of neurons in output layer = the number of classes in target column
])


# In[32]:


# compiling the neural network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[33]:


# training the neural network
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)


# In[34]:


# visualization accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.legend(['training data', 'validation data'], loc = 'lower right')


# In[35]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(['training data', 'validation data'], loc = 'upper right')


# In[36]:


# accuracy of the model on test data
loss, accuracy = model.evaluate(X_test_std, Y_test)
print(accuracy)


# In[37]:


print(X_test_std.shape)
print(X_test_std[0])


# In[38]:


Y_pred = model.predict(X_test_std)


# In[39]:


print(Y_pred.shape)
print(Y_pred[0])


# In[40]:


print(X_test_std)


# In[41]:


print(Y_pred)


# ### argmax funtion
# my_list = [0.25, 0.36]
# 
# index_of_max_value = np.argmax(my_list)
# 
# print(my_list)
# 
# print(index_of_max_value)
# 
# ### output
# 
# [0.25, 0.36]
# 
# 1

# In[42]:


# model.predict gives the prediction probability of each class for the data point

# converting the prediction probability to class labels
Y_pred_labels = [np.argmax(i) for i in Y_pred]
print(Y_pred_labels)


# In[43]:


# building the predictive system
input_data = (17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)
# change the input_data to a numpy array
input_data_as_np_array = np.asarray(input_data)
# reshape the numpy array as we are predicting one data point
input_data_reshaped = input_data_as_np_array.reshape(1, -1)
# standardizing the input data
input_data_std = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std)
print(prediction)

prediction_label = [np.argmax(prediction)]
print(prediction_label)

if (prediction_label[0] == 0):
    print("The tumor is Benign")
else:
    print("The tumor is Malignant")


# In[ ]:




