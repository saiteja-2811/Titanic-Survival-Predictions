# --------------------------------#
#   Titanic Disaster Predictions  #
# --------------------------------#

# We predict the outcome of the disaster for each Passenger ID ( Survived = 1 )

# ---------------------#
#   Import Libraries   #
# ---------------------#
import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score,accuracy_score
import xgboost as xgb
import sklearn.metrics as mt

# ------------------------------#
#   Exploratory Data Analysis   #
# ------------------------------#

#Importing the Data
train_data = pd.read_csv("data/train.csv")
# Get the profile report - saved this in the outputs folder as output.html
from pandas_profiling import ProfileReport
profile = ProfileReport(train_data,explorative=True)
profile.to_file('outputs/output.html')

# ------------------------#
#   Data Pre Processing   #
# ------------------------#

# Checking for Missing values in the data
train_data.isnull().sum()
# Dropping Cabin - Missing Values
train_data.drop(['Cabin'],axis=1,inplace=True)
# Dropping Fare - High Correlation with Pclass & Ambiguous Variable
train_data.drop(['Fare'],axis=1,inplace=True)
# Dropping Ticket - No value addition
train_data.drop(['Ticket'],axis=1,inplace=True)
# Missing Value Imputations
# Imputing Age with Median for a passenger class and gender
Age_Imp = train_data[['Pclass','Sex','Age']].dropna().groupby(by=['Pclass','Sex']).median().reset_index()
Age_Imp.rename(columns={'Age':'Age_Imp'},inplace=True)
train_data_v1 = pd.merge(train_data,Age_Imp,on=['Pclass','Sex'])
train_data_v1['Age'] = train_data_v1['Age'].fillna(train_data_v1['Age_Imp'])
train_data_v1.drop(['Age_Imp'],axis=1,inplace=True)
# Imputing Age with Mode
train_data_v1['Embarked'] = train_data_v1['Embarked'].fillna('S')

# ------------------------#
#   Feature Engineering   #
# ------------------------#
# Name Analysis
# Extract Mr / Mrs. / Master / Miss / Occupation info from Name
train_data_v1['Title'] = train_data_v1['Name'].str.split(',',n = 1, expand = True)[1].str.split(".",n = 1, expand = True)[0].str.strip()
train_data_v1.drop(['Name'],axis=1,inplace=True)
# Calculating Event Rates for Categories
Categ_List = ['Pclass','Sex','SibSp','Parch','Embarked','Title']
for var in Categ_List:
    Pvt = pd.DataFrame(pd.pivot_table(train_data_v1,values=['Survived'],index=var,aggfunc=['count','sum'],margins=True))
    Pvt['%Survived'] = Pvt['sum']['Survived']/Pvt['count']['Survived']
    print("\n",20*"*",var,"*"*20)
    print(Pvt['%Survived'])
# Combine Titles - Based on Event Rates
train_data_v1['Title'] = train_data_v1['Title'].replace(['Capt','Col','Don','Dr','Jonkheer','Lady','Major','Rev','Sir','the Countess'],'Executive')
train_data_v1['Title'] = train_data_v1['Title'].replace('Mlle','Miss')
train_data_v1['Title'] = train_data_v1['Title'].replace('Mme','Mrs')
train_data_v1['Title'] = train_data_v1['Title'].replace('Ms','Miss')

# Combine Sibsp & Parch - Based on Event Rates
train_data_v1.loc[(train_data_v1['SibSp'] + train_data_v1['Parch']) == 0,'Family'] = 'Alone'
train_data_v1.loc[(((train_data_v1['SibSp'] + train_data_v1['Parch']) > 0) & ((train_data_v1['SibSp'] + train_data['Parch']) <=4)),'Family'] = 'Small Family'
train_data_v1.loc[(train_data_v1['SibSp'] + train_data_v1['Parch']) > 4,'Family'] = 'Big Family'
train_data_v1.drop(['SibSp','Parch'],axis=1,inplace=True)

# Calculating Event Rates for Categories - Grouped
Categ_List = ['Pclass','Sex','Family','Embarked','Title']
for var in Categ_List:
    Pvt = pd.DataFrame(pd.pivot_table(train_data_v1,values=['Survived'],index=var,aggfunc=['count','sum'],margins=True))
    Pvt['%Survived'] = Pvt['sum']['Survived']/Pvt['count']['Survived']
    print("\n",20*"*",var,"*"*20)
    print(Pvt['%Survived'])


# ------------------------#
#   Weight of Evidence    #
# ------------------------#
# Calculate Weight of Evidences for Categories

#Information Value
def calc_iv(df, feature, target):
    """
    Input:
      * df : pandas.DataFrame,
      * feature : independent variable,
      * target : dependent variable
      
    Output: 
      * iv: float,
      * data: pandas.DataFrame
    """

    lst = [] #Declare Empty List

    df[feature] = df[feature].fillna("NULL") #Handle missing categories

    # Loop for a single variable
    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature,                                                        # Variable
                    val,                                                            # Value
                    df[df[feature] == val].count()[feature],                        # All
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature],  # Good (think: Survived == 1)
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature]]) # Bad (think: Survived == 0)
    # Append everything to a dataframe
    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Goods', 'Bads'])

    # Calculate the WOE now
    """
    #---------------------#
    #   WOE & IV Example  #
    #---------------------#
    
    #-------------------#
    #   Total           #
    #-------------------#                       
    Total Rows = 1000 || Total Goods = 300 || Total Bads = 700
    
    #-------------------#                  |       #-------------------#
    #   Category 1      #                  |       #   Category 2      #
    #-------------------#                  |       #-------------------# 
    All = 610 || Goods = 160 || Bads = 450 | All = 390 || Goods = 140 || Bads = 250
    
    #--------------#
    #   WOE & IV 1 #
    #--------------#    
    %Goods = (160/300) & %Bads = (450/700) => WOE 1 = ln(%Goods / %Bads) => IV 1 = WOE 1 * (%Goods - %Bads)
    
    #--------------#
    #   WOE & IV 2 #
    #--------------#    
    %Goods = (140/300) & %Bads = (250/700) => WOE 2 = ln(%Goods / %Bads) => IV 2 = WOE 2 * (%Goods - %Bads)
    
    """
    
    data['%Goods'] = data['Goods'] / data['Goods'].sum() # %Goods Calculation
    data['%Bads'] = data['Bads'] / data['Bads'].sum() # %Bads Calculation
    data['WoE'] = np.log(data['%Goods'] / data['%Bads']) # WOE Calculation
    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}}) #Exception handling

    data['IV'] = data['WoE'] * (data['%Goods'] - data['%Bads']) # IV Calculation
    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))
    data = data.drop(['Variable', 'All', 'Goods', 'Bads','%Goods','%Bads','IV'], axis=1)
    return data

#WOE for multiple features
def woe_imp(df1, feature, target, woe_feature):
    df2 = calc_iv(df1, feature, target)
    df1 = df1.merge(df2, left_on= feature, right_on='Value', how='left')
    df1 = df1.rename(columns = {"WoE": woe_feature,"Value":(str("Categ_") + feature)})
    return df1

# Imputing WOE for the Categorical Variables - Train Data
WOE_Out_Df = pd.DataFrame()
Categ_List = ['Pclass','Family','Embarked','Title']
for i in Categ_List:
    Var_Str = "WOE_" + str(i)
    train_data_v1 = woe_imp(train_data_v1, i, 'Survived', Var_Str)
    WOE_Imp_Df = train_data_v1[[Var_Str,(str("Categ_") + i)]].drop_duplicates(inplace=False).T.drop_duplicates().T
    WOE_Out_Df = pd.concat([WOE_Out_Df,WOE_Imp_Df])
    train_data_v1.drop([i,(str("Categ_") + i)],axis=1,inplace=True)
# Map Age
train_data_v1['Sex'] = train_data_v1['Sex'].map({'female':1,'male':0})

# --------------#
#   Modeling    #
# --------------#

# Split the data into train and test
from sklearn.model_selection import train_test_split
X = train_data_v1.iloc[:,2:]
y = train_data_v1.iloc[:,1:2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# Training the models

# Logit model
log_model = LogisticRegression()
log_result = log_model.fit(X_train,y_train)

# SVM Model
SVMmodel = LinearSVC()
SVM_result = SVMmodel.fit(X_train, y_train)

# DT Model
DTmodel = DecisionTreeClassifier()
DT_result = DTmodel.fit(X_train, y_train)

# RF Model
RFmodel = RandomForestClassifier(n_estimators = 500, #Trees in => Inc Accuracy
                                 max_depth=20, #Maximum depth of the tree
                                 bootstrap=True, #Pick different samples
                                 random_state=0) #Fix random state
RF_result = RFmodel.fit(X_train,y_train)

# LGB Model
LGBmodel = LGBMClassifier(objective= 'binary',
                          metric = 'accuracy',
                          subsample = 0.001,
                          num_leaves = 500,
                          max_depth=30,
                          num_iterations =500,
                          learning_rate = 0.01,
                          random_state=0)
LGB_result = LGBmodel.fit(X_train, y_train)

# XGB Model
XGBmodel = xgb.XGBClassifier(base_score=0.4, 
                             booster='gbtree',
                             gamma=0.2,
                             learning_rate=0.01,
                             max_depth=12,
                             min_child_weight=1,
                             n_estimators=300,
                             objective='binary:logistic',
                             random_state=0,
                             scale_pos_weight=1,
                             subsample=0.03,
                             verbosity=0,
                             eval_metric='merror')
XGB_result = XGBmodel.fit(X_train,y_train)

# Validating the models
log_yhat = log_result.predict(X_test)
svm_yhat = SVM_result.predict(X_test)
DT_yhat = DT_result.predict(X_test)
RF_yhat = RF_result.predict(X_test)
LGB_yhat = LGB_result.predict(X_test)
XGB_yhat = XGB_result.predict(X_test)

# Accuracy Calculation - Validation Data
acc_logit = accuracy_score(y_test, log_yhat)
acc_svm = accuracy_score(y_test,svm_yhat)
acc_DT = accuracy_score(y_test,DT_yhat)
acc_RF = accuracy_score(y_test,RF_yhat)
acc_LGB = accuracy_score(y_test,LGB_yhat)
acc_XGB = accuracy_score(y_test,XGB_yhat)

print("Logit model validation Accuracy: {:.2f}%".format(acc_logit*100))
print("SVM model validation Accuracy: {:.2f}%".format(acc_svm*100))
print("DT model validation Accuracy: {:.2f}%".format(acc_DT*100))
print("RF model validation Accuracy: {:.2f}%".format(acc_RF*100))
print("LGB model validation Accuracy: {:.2f}%".format(acc_LGB*100))
print("XGB model validation Accuracy: {:.2f}%".format(acc_XGB*100))

# -----------------------#
#   Scoring the test     #
# -----------------------#

# ----------------------------------------------------------#
#   Repeating all the Data Processing for the test data     #
# ----------------------------------------------------------#
#Importing the Data
test_data = pd.read_csv("data/test.csv")
# Dropping Cabin - Missing Values
test_data.drop(['Cabin'],axis=1,inplace=True)
# Dropping Fare - High Correlation with Pclass & Ambiguous Variable
test_data.drop(['Fare'],axis=1,inplace=True)
# Dropping Ticket - No Significance
test_data.drop(['Ticket'],axis=1,inplace=True)

# Missing Value Imputations
# Imputing Age with Median
Age_Imp_2 = test_data[['Pclass','Sex','Age']].dropna().groupby(by=['Pclass','Sex']).median().reset_index()
Age_Imp_2.rename(columns={'Age':'Age_Imp'},inplace=True)
test_data_v1 = pd.merge(test_data,Age_Imp_2,on=['Pclass','Sex'],how='left')
test_data_v1['Age'] = test_data_v1['Age'].fillna(test_data_v1['Age_Imp'])
test_data_v1.drop(['Age_Imp'],axis=1,inplace=True)
# Imputing Age with Mode
test_data_v1['Embarked'] = test_data_v1['Embarked'].fillna('S')

# Name Analysis
# Extract Mr / Mrs. / Master / Miss / Occupation info from Name
test_data_v1['Title'] = test_data_v1['Name'].str.split(',',n = 1, expand = True)[1].str.split(".",n = 1, expand = True)[0].str.strip()
test_data_v1.drop(['Name'],axis=1,inplace=True)

# Feature Engineering
# Combine Titles - Based on Event Rates
test_data_v1['Title'] = test_data_v1['Title'].replace(['Capt','Col','Don','Dona','Dr','Jonkheer','Lady','Major','Rev','Sir','the Countess'],'Executive')
test_data_v1['Title'] = test_data_v1['Title'].replace('Mlle','Miss')
test_data_v1['Title'] = test_data_v1['Title'].replace('Mme','Mrs')
test_data_v1['Title'] = test_data_v1['Title'].replace('Ms','Miss')

# Combine Sibsp & Parch - Based on Event Rates
test_data_v1.loc[(test_data_v1['SibSp'] + test_data_v1['Parch']) == 0,'Family'] = 'Alone'
test_data_v1.loc[(((test_data_v1['SibSp'] + test_data_v1['Parch']) > 0) & ((test_data_v1['SibSp'] + test_data_v1['Parch']) <=4)),'Family'] = 'Small Family'
test_data_v1.loc[(test_data_v1['SibSp'] + test_data_v1['Parch']) > 4,'Family'] = 'Big Family'
test_data_v1.drop(['SibSp','Parch'],axis=1,inplace=True)

# Imputing WOE for the Categorical Variables - Test Data
Categ_List = ['Pclass','Family','Embarked','Title']
for i in Categ_List:
    Var1 = str("Categ_"+i)
    Var2 = str("WOE_"+i)
    Mrgdf1 = WOE_Out_Df[[Var1,Var2]].drop_duplicates().dropna()
    test_data_v1 = pd.merge(test_data_v1,Mrgdf1,left_on = i,right_on=Var1,how='left')
    test_data_v1.drop([i,Var1],axis=1,inplace=True)
# Mapping the gender
test_data_v1['Sex'] = test_data_v1['Sex'].map({'female':1,'male':0})
# Change the datatypes
for i in ['WOE_Family','WOE_Embarked','WOE_Title']:
    test_data_v1[i] = test_data_v1[i].astype(float)
# Sort the dataframe
test_data_v1.sort_values(by=['PassengerId'],inplace=True)
test_data_y = pd.DataFrame(LGB_result.predict(test_data_v1.iloc[:,1:]))
Out = pd.merge(test_data_y,test_data_v1,left_index = True,right_index=True,how='left')
Out = Out.rename(columns={0:'Survived'})

# # Final Predictions
# ------------------------#
#   Final predictions     #
# ------------------------#
Out[['PassengerId','Survived']].to_csv("outputs/LGB_Submission.csv",index=False)
