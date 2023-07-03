#!/usr/bin/env python
# coding: utf-8

# ## Iris Flower Classification ML Project

# In this project, the classification of iris flower is done according to the given features. The features are:
# 
#     1) Sepal Length (in cm)
#     2) Petal Length (in cm)
#     3) Petal Length (in cm)
#     4) Petal Width (in cm)
# 
# The Iris Flower are of 3 types in this dataset.
# 
#     1) Iris Setosa
#     2) Iris Versicolor
#     3) Iris Virginica
# 
# 
# Different ML Classification Algorithms are used in this project like:
# 
#     1) Logistic Regression
#     2) Random Forest
#     3) Decision Trees
#     4) Support Vector Machines
#     5) K - Nearest Neighbours
# 

# In[57]:


import warnings
warnings.filterwarnings('ignore')
# Ignoring the warnings that might come in the process


# In[58]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Importing necessary libraries


# ### Data Importing and Information

# In[59]:


columns = ["Sepal Length","Sepal Width","Petal Length","Petal Width", "Class Label"]
# Naming the columns


# In[60]:


# Importing the .csv file with columns as the column names
iris = pd.read_csv('./iris.csv', names = columns)


# In[61]:


# Printing first 5 rows of data
iris.head()


# In[62]:


# Shape of dataset
iris.shape


# In[63]:


# More information about the dataset
iris.info()


# In[64]:


# More insights from the data
iris.describe()


# In[65]:


# Count of 3 classes of the flower
counts = iris["Class Label"].value_counts()
counts


# ## Data Preprocessing

# In[66]:


# Checking for missing values
iris.isna().sum()


# Since, the data has no missing values, for now no preprocessing is required.

# ## Exploratory Data Analysis

# In[67]:


# Setting theme of seaborn
sns.set_style("darkgrid")


# In[68]:


# Count the number of samples for each species
counts = iris["Class Label"].value_counts()

# Define the labels for the pie chart
labels = counts.index.tolist()

# Define the sizes for the pie chart
sizes = counts.tolist()

# Define the colors for the pie chart
colors = ["#E15759", "#4E79A7", "#76B7B2"]

# Create the pie chart using matplotlib
fig, ax = plt.subplots(figsize=(4, 4))
ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 14})
ax.axis('equal')

# Set the plot title
ax.set_title("Proportions of Iris Species", fontsize=16)
sns.despine()

# Show the plot
plt.show()


# There are equal number of examples of all 3 classes in the dataset (i.e. 50 each)

# In[69]:


sns.histplot(data = iris , x= "Sepal Length", bins = 20, hue = "Class Label")
plt.show()


# In[70]:


sns.histplot(data = iris , x= "Sepal Width", bins = 20,  hue = "Class Label")
plt.show()


# These two barplots look like a normal distrubution which means the model can quickly learn from the data.

# In[71]:


sns.histplot(data = iris , x= "Petal Length", bins = 20, hue = "Class Label")
plt.show()


# In[72]:


sns.histplot(data = iris , x= "Petal Width",bins = 20,  hue = "Class Label")
plt.show()


# These two histplots shows that the setosa has a comparatively smaller petal length and width than the other two.

# In[73]:


sns.relplot(data = iris, x="Sepal Length", y="Sepal Width", hue="Class Label", kind="scatter")
plt.title("Sepal Length vs. Sepal Width", fontsize=16)
plt.show()


# In[74]:


sns.relplot(data = iris, x="Petal Length", y="Petal Width", hue="Class Label", kind="scatter")
plt.title("Sepal Length vs. Sepal Width", fontsize=16)
plt.show()


# In[75]:


sns.relplot(data = iris, x="Sepal Length", y="Petal Length", hue="Class Label", kind="scatter")
plt.title("Sepal Length vs. Sepal Width", fontsize=16)
plt.show()


# In[76]:


sns.relplot(data = iris, x="Sepal Width", y="Petal Width", hue="Class Label", kind="scatter")
plt.title("Sepal Length vs. Sepal Width", fontsize=16)
plt.show()


# From all these scatterplots, we can observe that the setosa class is showing very drastic change in properties.

# In[77]:


# Pair Plot between all students
sns.pairplot(iris, hue = "Class Label")
plt.show()


# In[78]:


# Finding correlation between all variables
corr = iris.corr()
fig, ax = plt.subplots(figsize =(5,4))
sns.heatmap(corr, annot = True, ax = ax)
plt.show()


# From this heatmap, there is a strong positive relationship between sepal length and the petal dimensions whereas strong negative relationship between sepal width and the petal dimensions.

# ### Label Encoder

# Changing the categorical value into integer.

# In[79]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Changing the class label into integer values (0,1,2)
iris['Class Label'] = le.fit_transform(iris['Class Label'])
iris.head()


# In[80]:


iris['Class Label'].unique()


# ### Seperating Data to features and output

# In[81]:


# Seperating Input and output data (X and y)
data = iris.values
X = data[:, 0:4]
y = data[:, 4]


# ### Train Test Split

# In[82]:


# Splitting the data in train and test state
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4 , random_state=42 , shuffle = True)


# In[83]:


X_train.shape


# In[84]:


X_test.shape


# In[85]:


y_train = y_train.astype(int)
y_train.shape


# In[86]:


y_test = y_test.astype(int)
y_test.shape


# ### Scaling the data

# In[87]:


# Scaling the data between some limits
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# X_test


# ### Metrics

# In[88]:


# Importing error and accutacy metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ## Model Training

# ### Logistic Regression

# In[89]:


from sklearn.linear_model import LogisticRegression

logreg_model = LogisticRegression(random_state= 0)
logreg_model.fit(X_train, y_train)

ypred_logreg = logreg_model.predict(X_test)
ypred_logreg


# In[90]:


print(classification_report(y_test, ypred_logreg))
# Classification Report


# In[91]:


accuracy_score(y_test, ypred_logreg)


# ### Random Forest Classifier

# In[92]:


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train,y_train)

ypred_rf = rf_model.predict(X_test)
ypred_rf


# In[93]:


print(classification_report(y_test, ypred_rf))


# In[94]:


accuracy_score(y_test, ypred_rf)


# ### Decision Trees

# In[95]:


from sklearn.tree import DecisionTreeClassifier

dtree_model = DecisionTreeClassifier()
dtree_model.fit(X_train, y_train)

ypred_dtree = dtree_model.predict(X_test)


# In[96]:


print(classification_report(y_test, ypred_dtree))


# In[97]:


accuracy_score(y_test, ypred_dtree)


# In[98]:


confusion_matrix(y_test, ypred_dtree)


# ### Support Vector Machines

# In[99]:


from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)

ypred_svc = svc_model.predict(X_test)


# In[100]:


print(classification_report(y_test, ypred_svc))


# In[101]:


accuracy_score(y_test, ypred_svc)


# In[102]:


confusion_matrix(y_test, ypred_svc)


# ### K - Nearest Neighbours

# In[103]:


from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

ypred_knn = knn_model.predict(X_test)


# In[104]:


accuracy_score(y_test, ypred_knn)


# In[105]:


print(classification_report(y_test, ypred_knn))


# In[106]:


confusion_matrix(y_test, ypred_knn)


# In[107]:


# Accuracy from different Models
print("Accuracy of Logistic Regression Model:",accuracy_score(y_test, ypred_logreg)*100)
print("Accuracy of Random Forest Classification Model:",accuracy_score(y_test,ypred_rf)*100)
print("Accuracy of Decision Tree Model:",accuracy_score(y_test, ypred_dtree)*100)
print("Accuracy of SVM Model:",accuracy_score(y_test,ypred_svc)*100)
print("Accuracy of KNN Model:",accuracy_score(y_test,ypred_knn)*100)


# ### A random data?

# In[108]:


x_sample = [[4.5,3.1,2.3,3.4]]


# In[109]:


# Using Random Forest Model to predict a random input data
y_predicted = rf_model.predict(x_sample)
y_predicted # Versicolor


# In this way, we successfully trained a Iris Flower Classification ML Project using many Classification Models.
