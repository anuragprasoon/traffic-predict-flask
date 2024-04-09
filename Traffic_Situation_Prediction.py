#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.pyplot as plotter
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.model_selection import KFold
from scipy import stats
from scipy.stats import norm, skew 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')


# #          Read Dataset

# In[2]:


train = pd.read_csv('Traffic.csv')
train 


# In[3]:


train, test = train_test_split(train,test_size=0.1,random_state=1992)
print("Shape of train: ",train.shape)
print("Shape of test",test.shape)


# # Visualization
# 

# In[4]:


train.isnull().sum()


# In[5]:


test.isnull().sum()


# In[6]:





# In[7]:


sns.histplot(train,x='Day of the week',hue='Traffic Situation',kde=True)


# In[8]:


sns.histplot(train,x='Date',hue='Traffic Situation',kde=True)


# In[9]:


sns.histplot(train,x='CarCount',hue='Traffic Situation',kde=True)


# In[10]:


sns.histplot(train,x='BikeCount',hue='Traffic Situation',kde=True)


# In[11]:


sns.histplot(train,x='BusCount',hue='Traffic Situation',kde=True)


# In[12]:


sns.histplot(train,x='TruckCount',hue='Traffic Situation',kde=True)


# In[13]:


sns.histplot(train,x='Total',hue='Traffic Situation',kde=True)


# # preprocessing

# In[14]:


plt.subplot(1, 3, 1)
sns.countplot(x = train["Day of the week"])
plotter.xticks(rotation = 90);

plt.subplot(1, 3, 3)
sns.countplot(x = train["Traffic Situation"])
plotter.xticks(rotation = 90);
plt.show()


# In[15]:


sns.histplot(train,x='Time',hue='Traffic Situation',kde=True)


# In[16]:


df_temp=train
df_temp['Day of the week'] = df_temp['Day of the week'].replace({'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday': 6,'Sunday':7})
df_temp['Traffic Situation'] = df_temp['Traffic Situation'].replace({'low': 0,'normal': 1,'high': 2, 'heavy':3})
train=df_temp
train


# # feature selection

# In[17]:


train_temp=train

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_temp['Time'] = le.fit_transform(train_temp['Time'])

corr = train_temp.corr(method='pearson')
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f")
plt.xticks(range(len(corr.columns)), corr.columns);
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()


# In[18]:


from xgboost import XGBClassifier
X_data_feature= train.drop(columns=['Traffic Situation'],axis=1)
y_data_feature= train['Traffic Situation']

model = [XGBClassifier()]

model = [model[i].fit(X_data_feature,y_data_feature) for i in range(len(model))]

num_chr = [12, 12, 10]

for i in range(len(model)):
    print(str(model[i])[:num_chr[i]] + ': \n',
          model[i].feature_importances_)
    feat_importances = pd.Series(model[i].feature_importances_,
                                 index=X_data_feature.columns)
    feat_importances.nlargest(10).plot(kind='barh', color='royalblue')
    plt.xlim(0, 1.0)
    plt.show()


# In[ ]:





# In[19]:


import lightgbm as lgb
from lightgbm import LGBMClassifier

model = [LGBMClassifier()]

model = [model[i].fit(X_data_feature,y_data_feature) for i in range(len(model))]

num_chr = [12, 12, 10]

for i in range(len(model)):
    print(str(model[i])[:num_chr[i]] + ': \n',
          model[i].feature_importances_)
    feat_importances = pd.Series(model[i].feature_importances_,
                                 index=X_data_feature.columns)
    feat_importances.nlargest(10).plot(kind='barh', color='royalblue')
    plt.xlim(0, 4500)
    plt.show()


# In[20]:


from catboost import CatBoostClassifier

model = [CatBoostClassifier(logging_level='Silent')]

model = [model[i].fit(X_data_feature,y_data_feature) for i in range(len(model))]

num_chr = [12, 12, 10]

for i in range(len(model)):
    print(str(model[i])[:num_chr[i]] + ': \n',
          model[i].feature_importances_)
    feat_importances = pd.Series(model[i].feature_importances_,
                                 index=X_data_feature.columns)
    feat_importances.nlargest(10).plot(kind='barh', color='royalblue')
    plt.xlim(0, 40)
    plt.show()


# In[21]:


train = train.drop(columns=["Day of the week","Date"],axis=1)
train


# In[22]:


X= train.drop(columns=["Traffic Situation"],axis=1)
y= train["Traffic Situation"]


# In[23]:


X_train=X
y_train=y

from sklearn.preprocessing import StandardScaler
StandardScaler = StandardScaler()  
X_train = StandardScaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train)
X_train


# # target distribution
# 

# In[24]:


sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(3, 4))
sns.histplot(y_train)
ax.xaxis.grid(False)

sns.despine(trim=True, left=True)
plt.show()

print("Skewness: %f" % y_train.skew())
print("Kurtosis: %f" % y_train.kurt())


# # Split Dataset

# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train,test_size=0.2,random_state=2019)
print("Shape of X_train: ",X_train.shape)
print("Shape of X_eval: ", X_eval.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_eval",y_eval.shape)


# In[26]:


y_train =pd.DataFrame(y_train)
y_eval =pd.DataFrame(y_eval)


# # VotingClassifier

# In[27]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier,RidgeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier,BaggingClassifier
from sklearn.ensemble import  AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC


# In[28]:


from sklearn.ensemble import VotingClassifier

clf1 = XGBClassifier()
clf2 = RandomForestClassifier()
clf3 = ExtraTreesClassifier()
clf4 = CatBoostClassifier(logging_level='Silent')
clf5 = KNeighborsClassifier()
clf6 = LogisticRegression()
clf7=  RidgeClassifier()
clf8= HistGradientBoostingClassifier()
clf9= BaggingClassifier()
clf10= GradientBoostingClassifier()
clf11= GaussianNB()
clf12= LGBMClassifier()


eclf = VotingClassifier(estimators=[ ('XGB', clf1), ('RF', clf2), ('ET', clf3), ('CAT', clf4), ('KN', clf5),
                                   ('LG', clf6), ('RC', clf7), ('HBC', clf8), ('BC', clf9), ('GBC', clf10), ('GNB', clf11), 
                                    ('LGBM', clf12)],voting='hard')

for clf, label in zip([clf1,clf2,clf3,clf4,clf5,clf6,clf7,clf8,clf9,clf10,clf11,clf12], 
                      ['XGB','RF','ET','CAT','KN','LG','RC','HBC','BC','GBC','GNB','LGBM']):
    scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


# # StackingClassifier

# In[29]:


class StackingAveragedModels(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


# In[30]:


stacked_averaged_models = StackingAveragedModels(base_models = (clf2,clf9,clf10,clf12),meta_model = clf4)


# In[31]:


stacking_model=stacked_averaged_models.fit(X_train.values, y_train.values)


# In[32]:


stacking_model.fit(X_train.values , y_train.values)
y_pred_stacking = stacking_model.predict(X_eval.values) 
stacking_acc = accuracy_score(y_eval.values, y_pred_stacking)
print("stacking accuracy is: {0:.3f}%".format(stacking_acc * 100))
cm = confusion_matrix(y_eval, y_pred_stacking)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='.0f')
plt.xlabel("Predicted Digits")
plt.ylabel("True Digits")
plt.show()


# # Predict Test Data

# In[33]:


test = test.reset_index(drop=True)
test_temp=test


# In[34]:


df_temp=test_temp
df_temp['Day of the week'] = df_temp['Day of the week'].replace({'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday': 6,'Sunday':7})
df_temp['Traffic Situation'] = df_temp['Traffic Situation'].replace({'low': 0,'normal': 1,'high': 2, 'heavy':3})
test_temp=df_temp
test_temp


# In[35]:


test_temp = test_temp.drop(columns=['Traffic Situation'],axis=1)
test_temp


# In[36]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
test_temp['Time'] = le.fit_transform(test_temp['Time'])
test_temp=test_temp.drop(columns=['Day of the week',"Date"],axis=1)
test_temp


# In[37]:


test_row = test_temp.shape[0]
test_row 


# In[38]:


import_train = X.reset_index(drop=True)
import_train


# In[39]:


Row_Number=test_row 
X_test_target1_df=import_train._append(test_temp,ignore_index=True)

from sklearn.preprocessing import StandardScaler
StandardScaler = StandardScaler()  
X_test_target1_df = StandardScaler.fit_transform(X_test_target1_df)
test_pred_target0= pd.DataFrame(X_test_target1_df)
test_pred_target0 = pd.DataFrame(test_pred_target0).tail(Row_Number)
test_pred_target0 = test_pred_target0.reset_index(drop=True)
test_pred_target0


# In[40]:


test_pred_target0.isnull().sum()


# In[41]:


Stacking_predict=stacking_model.predict(test_pred_target0.values)


# In[42]:


#DataFrame
Stacking_predict_df=pd.DataFrame(Stacking_predict)

#rename lable
Stacking_predict_df=Stacking_predict_df.set_axis(axis=1,labels=['Stack_pred'])

#merge predict
test_pred=test.merge(Stacking_predict_df,how='inner',left_index=True,right_index=True)
test_pred


# In[43]:


stacking_acc = accuracy_score(test_pred['Traffic Situation'], test_pred['Stack_pred'])
print("stacking accuracy is: {0:.3f}%".format(stacking_acc * 100))
cm = confusion_matrix(test_pred['Traffic Situation'], test_pred['Stack_pred'])
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='.0f')
plt.xlabel("Predicted Digits")
plt.ylabel("True Digits")
plt.show()


# # Predictor 

# ###  Input time, carcount,bikecount,buscount,truck count,totall

# In[44]:


Time_in_24_hr = int(input())
CarCount = int(input())
BikeCount = int(input())
BusCount = int(input())
TruckCount = int(input())
Total = int(input())
#90 82 16 14 34 146
#63 112 13 37 6 168


# In[46]:


arr = np.array([[Time_in_24_hr,CarCount,BikeCount,BusCount,TruckCount,Total]])
columns = ['Time','CarCount', 'BikeCount', 'BusCount', 'TruckCount','Total']
df = pd.DataFrame(arr, columns=columns)

Row_Number=1
X_test_target1_df1=import_train._append(df,ignore_index=True)

from sklearn.preprocessing import StandardScaler
StandardScaler = StandardScaler()  
X_test_target1_df1 = StandardScaler.fit_transform(X_test_target1_df1)
test_pred_target01= pd.DataFrame(X_test_target1_df1)
test_pred_target01 = pd.DataFrame(test_pred_target01).tail(Row_Number)
test_pred_target01 = test_pred_target01.reset_index(drop=True)
print(test_pred_target01)


y_pred1=stacking_model.predict(test_pred_target01) 
print(y_pred1)


# In[ ]:




