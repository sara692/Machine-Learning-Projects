#!/usr/bin/env python
# coding: utf-8

# ## Initial Setting 

# In[138]:


# 📊 DATA MANIPULATION & ANALYSIS LIBRARIES
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, pearsonr

# 📈 DATA VISUALIZATION LIBRARIES
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 🤖 MACHINE LEARNING & MODELING LIBRARIES
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, confusion_matrix, 
    classification_report, roc_auc_score, silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 🔧 UTILITY & SYSTEM LIBRARIES
import os
import warnings
from datetime import datetime
import re
from collections import Counter

# 🎨 VISUALIZATION SETUP
# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Plotly settings
import plotly.io as pio
pio.templates.default = "plotly_white"

# 📏 DISPLAY OPTIONS
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

# ⚠️ WARNINGS CONFIGURATION
warnings.filterwarnings('ignore')


# ## Importing The data

# In[78]:


try:
    df = pd.read_csv(r"C:\Users\v\Downloads\archive\bi.csv", encoding='latin-1')
    print("✅ Loaded with latin-1 encoding!")
except:
    pass


# ## Data Cleaning and Preprocessing

# In[79]:


df.info()


# In[80]:


df.describe()


# In[81]:


df.isnull().sum()


# In[82]:


df.duplicated().sum()


# In[83]:


df["Python"] = df["Python"].fillna(df["Python"].mean())
df.isnull().sum()


# In[84]:


string_columns = df.select_dtypes(include=['object']).columns.tolist()

print("🔍 UNIQUE VALUES IN STRING COLUMNS")
print("=" * 50)

for col in string_columns:
    unique_values = df[col].unique()
    print(f"\n📊 {col}:")
    print(f"   Unique count: {len(unique_values)}")
    print(f"   Values: {list(unique_values)}")


# In[85]:


df['gender'] = df['gender'].replace({
    'M': 'Male', 'F': 'Female', 'male': 'Male', 'female': 'Female'
})


# In[86]:


df['country'] = df['country'].replace({
    'Norway': 'Norway', 'Norge': 'Norway', 
    'Rsa': 'South Africa', 'UK': 'United Kingdom',
    'Somali': 'Somalia'
})


# In[87]:


df['residence'] = df['residence'].replace({
    'BI-Residence': 'BI Residence',
    'BIResidence': 'BI Residence',
    'BI_Residence': 'BI Residence'
})


# In[88]:


df['prevEducation'] = df['prevEducation'].replace({
    'Barrrchelors': 'Bachelors',
    'Diplomaaa': 'Diploma',
    'diploma': 'Diploma',
    'DIPLOMA': 'Diploma',
    'HighSchool': 'High School'
})


# In[89]:


string_columns = df.select_dtypes(include=['object']).columns.tolist()

print("🔍 UNIQUE VALUES IN STRING COLUMNS")
print("=" * 50)

for col in string_columns:
    unique_values = df[col].unique()
    print(f"\n📊 {col}:")
    print(f"   Unique count: {len(unique_values)}")
    print(f"   Values: {list(unique_values)}")


# In[90]:


df.columns = df.columns.str.title()
df = df.rename(columns={
    'Fname': 'FName',
    'Lname': 'LName',
    'Entryexam': 'Entry_Exam',
    'Preveducation': 'Prev_Education',
    'Studyhours': 'Study_Hours'
    
})
df.columns


# ## Exploratory Data Analysis

# In[91]:


df.hist(figsize=(10,8))
plt.suptitle('Histogram of Variables', fontsize=14, fontweight='bold')
plt.tight_layout()  
plt.show()


# In[92]:


df.boxplot(figsize=(10, 8), color='royalblue')
plt.title('Outliers Distribution', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[105]:


df_numeric=df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
corr = df_numeric.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# In[107]:


corr


# ## Predictive Modeling

# In[126]:


le = LabelEncoder()

# Apply to categorical columns
df['Gender'] = le.fit_transform(df['Gender'])
df['Country'] = le.fit_transform(df['Country']) 
df['Prev_Education'] = le.fit_transform(df['Prev_Education'])
df['Residence'] = le.fit_transform(df['Residence'])

print("✅ Categorical columns converted to numbers!")
df


# In[128]:


df['Total_Percentage'] = ((df['Python'] + df['Db']) / 200) * 100
feature_columns = ['Age', 'Gender', 'Country', 'Residence', 'Entry_Exam', 'Prev_Education', 'Study_Hours']
X = df[feature_columns]
y = df['Total_Percentage']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    shuffle=True
)

print("✅ Data ready for total percentage prediction!")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train range: {y_train.min():.1f}% - {y_train.max():.1f}%")


# In[134]:


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_train


# In[135]:


def create_models():
    """
    Create multiple machine learning models for regression
    Returns: Dictionary of models
    """
    
    lr_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    knn_model = KNeighborsRegressor(n_neighbors=5)
    dt_model = DecisionTreeRegressor(random_state=42)
    
    # All models dictionary
    all_models = {
        "linear_regression": lr_model,
        "random_forest": rf_model,
        "knn": knn_model,
        "decision_tree": dt_model
    }
    
    return all_models


# In[194]:


def evaluate_models(models, X_train, X_test, y_train, y_test):
    
    results = {}
    
    print(f"\n MODEL EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Model':<20} {'MAE':<12} {'RMSE':<14} {'R2':<10}")
    print("-" * 80)
    
    for name, model in models.items():
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        MAE = mean_absolute_error(y_test, pred)
        MSE = mean_squared_error(y_test, pred)    # This is Mean Squared Error
        RMSE = np.sqrt(MSE)                       # This is Root Mean Squared Error
        R2 = r2_score(y_test, pred)
        
        results[name] = {
            "models": model,
            "MAE": MAE,
            "RMSE": RMSE,
            "R2": R2
        }
        print(f"{name:<20} {MAE:.2f}         {RMSE:.2f}          {R2:.2f} ")        
        best_model_name, best_model_values = max(results.items(),  key=lambda x:
                                                (x[1]["R2"],
                                                -x[1]["RMSE"],
                                                -x[1]["MAE"]))
    print("\n")
    print(f"⭐The Best Model : {best_model_name} → MAE: {best_model_values['MAE']:.2f}, "
    f"RMSE: {best_model_values['RMSE']:.2f}, "
    f"R2: {best_model_values['R2']:.2f}")
    
    return 


# In[195]:


models = create_models()
results = evaluate_models(models, X_train, X_test, y_train, y_test)
results


# In[ ]:




