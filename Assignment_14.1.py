
# coding: utf-8

# ## In this assignment students need to predict whether a person makes over 50K per year or not from classic adult dataset using XGBoost.

# In[1]:


#import libraries
import numpy as np
import pandas as pd


# In[2]:


#Loading dataset

train_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header = None)
test_set = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', skiprows = 1, header = None)


# In[3]:


# column names and assigning them to train and test set

col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status','occupation','relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week','native_country', 'wage_class']
train_set.columns = col_labels
test_set.columns = col_labels


# In[4]:


# test set first five rows
test_set.head()


# In[5]:


# to check statistics of numerical features.
train_set.describe()


# In[6]:


# created a column in train and test set train_id and test_id respectively.
train_set['train_ind'] = 1


# In[7]:


test_set['train_ind'] = 0


# In[8]:


#combined data 
combined_data = train_set.append(test_set)


# In[9]:


#size of combined data
combined_data.shape


# In[10]:


# Columns of the dataset
combined_data.describe(include = ['O']).columns


# #### in this dataset missing values are denoted as ? so inorder to determine number of nulls we convert ? with np.nan.

# In[11]:


df1 = combined_data.replace(' ?', np.nan)


# In[12]:


# columns and number of nulls in that columns
df1.isnull().sum()


# In[13]:


# Filling NaN with ' unknown'
df1.fillna(' unknown', inplace = True)


# In[14]:


# after fillna now there is no missing values in the features set.
df1.isnull().sum()


# Target variable in the dataset is a binary variable which will classify data into two categories: 1. one who earns equal to or more than 50 K and the one who don't. Here we have created a column as Target_variable and assign 0 to all the rows. then we assigned 1 to the rows where 'wage_class' has entry as ' >50K' or ' >50K.'. In this way we have two classes in the Target_variable: 1 for those with income 50K and more  and 0 for rest.

# In[15]:


df1['wage_class'].unique()


# In[16]:


df1['target_variable'] = 0


# In[17]:


df1.loc[df1['wage_class'] == ' >50K' ,'target_variable'] = 1


# In[18]:


df1.loc[df1['wage_class'] == ' >50K.' ,'target_variable'] = 1


# In[19]:


df1['target_variable'].value_counts()


# In[20]:


df1.shape


# In[21]:


df1.head()


# #### creating dummies for all the categorical variables. starting from relationship:

# In[22]:


df1['relationship'].unique()


# In[23]:


dummies = pd.get_dummies(df1['relationship'], prefix = 'relationship') # this command will create as many columns as many categories 
# in the dataset and new columns will have relationship as prefix.

#df1 = df1.join(dummies)


# In[24]:


df1 = pd.concat([df1,dummies],axis = 1)


# In[25]:


df1.shape


# In[26]:


dummies.shape


# In[27]:


#dropping original columns from the dataframe whose we have added dummies.
df1.drop('relationship', axis = 1, inplace = True)


# In[28]:


df1.head()


# In[29]:


df1.shape


# In[30]:


# performing one hot encoding on each categorical variable. Due to memory limitation can't put on loop.


# In[31]:


df1.describe(include = ['O']).columns


# In[32]:


dummies_workclass = pd.get_dummies(df1['workclass'], prefix = 'workclass')


# In[33]:


dummies_education = pd.get_dummies(df1['education'], prefix = 'education')


# In[34]:


dummies_marital_status = pd.get_dummies(df1['marital_status'], prefix = 'marital_status')


# In[35]:


dummies_occupation = pd.get_dummies(df1['occupation'], prefix = 'occupation')


# In[36]:


dummies_race = pd.get_dummies(df1['race'], prefix = 'race')


# In[37]:


dummies_sex = pd.get_dummies(df1['sex'], prefix = 'sex')


# In[38]:


df1 = pd.concat([df1,dummies_workclass,dummies_education,
                 dummies_marital_status,dummies_occupation,dummies_race,dummies_sex],axis = 1)


# ### Dropped redundant columns

# In[39]:


df1.drop(['workclass', 'education', 'marital_status', 'occupation', 'race', 'sex',
          'wage_class'], axis = 1, inplace = True)


# In[40]:


df1.shape


# In[41]:


df1['country']= 0


# In[42]:


df1.loc[df1['native_country'] == ' United-States' ,'country'] = 1


# In[43]:


df1['country'].value_counts()


# In[44]:


df1.drop('native_country', axis = 1, inplace = True)


# In[45]:


df1.head()


# In[46]:


import numpy as np, pandas as pd, matplotlib.pyplot as plt #, pydotplus
from sklearn import tree, metrics, model_selection, preprocessing
from IPython.display import Image, display


# In[47]:


final_train_set = df1[df1["train_ind"] == 1]


# In[48]:


final_train_set.shape


# In[49]:


final_test_set = df1[df1["train_ind"] == 0]


# In[50]:


final_test_set.shape


# In[51]:


# select features
y = final_train_set.pop('target_variable')


# In[52]:


y.shape


# In[53]:


X = final_train_set


# #### Train test split

# In[54]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)


# In[55]:


# train the decision tree
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
dtree.fit(X_train, y_train)


# In[56]:


# use the model to make predictions with the test data
y_pred = dtree.predict(X_test)


# In[57]:


# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


# In[58]:


#Import Xgboost
import xgboost as xgb


# In[59]:


# to feed data to xgboost first training set is transformed into Dmatrix. in code below train and test sets are transformed.

xgtrain = xgb.DMatrix(X_train, label = y_train)
xgtest = xgb.DMatrix(X_test, label = y_test)


# In[60]:


# to see out output 
watchlist = [(xgtrain,'train'),(xgtest, 'eval')]


# In[61]:


# parameters

params = {}
params["objective"] =  "binary:logistic"
params["booster"] = "gbtree"
params["max_depth"] = 7
params["eval_metric"] = 'auc'
params["subsample"] = 0.8
params["colsample_bytree"] = 0.8
params["silent"] = 1
params["seed"] = 4
params["eta"] = 0.1

plst = list(params.items())


# In[62]:


#Running the model with 15 iterations and parameters defined above

num_rounds = 150
model_cv = xgb.train(plst, xgtrain, num_rounds, evals = watchlist, early_stopping_rounds = 10, verbose_eval = True)


# In[63]:


### Clearly training AUC is 95% and validation AUC is 92%. Model does very well on unseen data as well.


# In[64]:


feat_imp = pd.Series(model_cv.get_fscore()).sort_values(ascending=False)


# In[65]:


import matplotlib.pyplot as plt
feat_imp[:25].plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()

