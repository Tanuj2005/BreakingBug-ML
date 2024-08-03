# FIXED IMPORT ERRORS 

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from yellowbrick.cluster import KElbowVisualizer
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv("PATH TO DATSET")
df.head()
df.info()
df.shape
df['id'].min(), df['id'].max()
df['age'].min(), df['age'].max()
df['age'].describe()

import seaborn as sns

custom_colors = ["#FF5733", "#3366FF", "#33FF57"] 

sns.histplot(df['age'], kde=True, color="#FF5733", palette=custom_colors)

sns.histplot(df['age'], kde=True)
plt.axvline(df['age'].mean(), color='Red')
plt.axvline(df['age'].median(), color= 'Green')
plt.axvline(df['age'].mode()[0], color='Blue')


print('Mean', df['age'].mean())
print('Median', df['age'].median())
print('Mode', df['age'].mode())

fig = px.histogram(data_frame=df, x='age', color= 'sex')
fig.show()

df['sex'].value_counts()

male_count = 726
female_count = 194

total_count = male_count + female_count

male_percentage = (male_count/total_count)*100
female_percentages = (female_count/total_count)*100

print(f'Male percentage i the data: {male_percentage:.2f}%')
print(f'Female percentage in the data : {female_percentages:.2f}%')

difference_percentage = ((male_count - female_count)/female_count) * 100
print(f'Males are {difference_percentage:.2f}% more than female in the data.')


726/194

df.groupby('sex')['age'].value_counts()

df['dataset'].count()

fig =px.bar(df, x='dataset', color='sex')
fig.show()

print (df.groupby('sex')['dataset'].value_counts())

fig = px.histogram(data_frame=df, x='age', color= 'dataset')
fig.show()

# FIXED BUGS IN THE PRINT STATEMENTS BELOW

print("___________________________________________________________")
print("Mean of the age column: ", df['age'].mean())
print("___________________________________________________________")
print("Median of the age column: ", df['age'].median())
print("___________________________________________________________")
print("Mode of the age column: ", df['age'].mode().iloc[0])  
print("___________________________________________________________")

df['cp'].value_counts()

sns.countplot(df, x='cp', hue= 'sex')
sns.countplot(df,x='cp',hue='dataset')

fig = px.histogram(data_frame=df, x='age', color='cp')
fig.show()
df['trestbps'].describe()

print(f"Percentage of missing values in trestbps column: {df['trestbps'].isnull().sum() /len(df) *100:.2f}%")

imputer1 = IterativeImputer(max_iter=10, random_state=42)

imputer1.fit(df[['trestbps']])

df['trestbps'] = imputer1.transform(df[['trestbps']])

print(f"Missing values in trestbps column: {df['trestbps'].isnull().sum()}")

df.info()

(df.isnull().sum()/ len(df)* 100).sort_values(ascending=False)

# FIXED THE IMPUTER ERROR 

imputer2 = IterativeImputer(max_iter=10, random_state=42)

df['ca'] = imputer2.fit_transform(df[['ca']])
df['oldpeak'] = imputer2.fit_transform(df[['oldpeak']])
df['chol'] = imputer2.fit_transform(df[['chol']])
df['thalch'] = imputer2.fit_transform(df[['thalch']])

# FIXED THE ISNULL CODE

(df.isnull().sum()/ len(df)* 100).sort_values(ascending=False)

print(f"The missing values in thal column are: {df['thal'].isnull().sum()}")

df['thal'].value_counts()

df.tail()

df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=True)

missing_data_cols = df.isnull().sum()[df.isnull().sum()>0].index.tolist()

missing_data_cols

cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols

Num_cols = df.select_dtypes(exclude='object').columns.tolist()
Num_cols

print(f'categorical Columns: {cat_cols}')
print(f'numerical Columns: {Num_cols}')

categorical_cols = ['thal', 'ca', 'slope', 'exang', 'restecg','thalch', 'chol', 'trestbps']
bool_cols = ['fbs']
numerical_cols = ['oldpeak','age','restecg','fbs', 'cp', 'sex', 'num']

passed_col = categorical_cols

# FIXED THE IMPUTE CATEGORICAL MISSING DATA FUNCTION

def impute_categorical_missing_data(df, passed_col, missing_data_cols):
    # Separate rows with and without missing values in the passed column
    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    # Feature matrix (X) and target vector (y) for rows without missing values in passed_col
    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]

    # Encode categorical features in X
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Apply LabelEncoder to categorical columns in X
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = label_encoder.fit_transform(X[col].astype(str))

    # Encode y if it is a categorical column
    if y.dtype == 'object':
        y = label_encoder.fit_transform(y.astype(str))

    # Imputation setup for other missing columns
    imputer = IterativeImputer(estimator=RandomForestClassifier(random_state=16), add_indicator=True)

    for col in missing_data_cols:
        if col != passed_col and X[col].dtype != 'object':
            X[col] = imputer.fit_transform(X[col].values.reshape(-1, 1))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Predict and calculate accuracy
    y_pred = rf_classifier.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)

    print(f"The feature '{passed_col}' has been imputed with {round(acc_score * 100, 2)}% accuracy\n")

    # Impute missing values in df_null
    if len(df_null) > 0:
        X_null = df_null.drop(passed_col, axis=1)

        # Apply LabelEncoder to categorical columns in X_null
        for col in X_null.columns:
            if X_null[col].dtype == 'object':
                X_null[col] = label_encoder.fit_transform(X_null[col].astype(str))

        df_null[passed_col] = rf_classifier.predict(X_null)

        # If the column was originally a categorical column, map back to original categories
        if y.dtype == 'object':
            df_null[passed_col] = label_encoder.inverse_transform(df_null[passed_col])

    # Combine the data
    df_combined = pd.concat([df_not_null, df_null])

    return df_combined[passed_col]

#  FIXED THE IMPUTE CONTINUOUS MISSING DATA FUNCTION

def impute_continuous_missing_data(df, passed_col, missing_data_cols):
    # Split into rows with and without missing values in the passed column
    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    # Feature matrix (X) and target vector (y) for rows without missing values in passed_col
    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]

    # Ensure all data in X is numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Imputation using IterativeImputer (if needed for other columns)
    imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=16), add_indicator=True)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForestRegressor
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_regressor.fit(X_train, y_train)

    # Prediction
    y_pred = rf_regressor.predict(X_test)

    # Performance Metrics
    print("MAE =", mean_absolute_error(y_test, y_pred))
    print("RMSE =", mean_squared_error(y_test, y_pred, squared=False))
    print("R2 =", r2_score(y_test, y_pred))

    # If there are missing values, impute them
    if len(df_null) > 0:
        X_null = df_null.drop(passed_col, axis=1)
        X_null = X_null.apply(pd.to_numeric, errors='coerce')  # Ensure numeric data in X_null
        df_null[passed_col] = rf_regressor.predict(X_null)
    
    # Combine imputed rows with the original non-missing rows
    df_combined = pd.concat([df_not_null, df_null])

    return df_combined[passed_col]

df.isnull().sum().sort_values(ascending=False)

import warnings
warnings.filterwarnings('ignore')

# FIXED THE BELOW CODE BELOW FOR CALLING THE ABOVE FUNSTIONS

for col in missing_data_cols:
    print("Missing Values", col, ":", str(round((df[col].isnull().sum() / len(df)) * 100, 2))+"%")
    if col in categorical_cols:
        df[col] = impute_categorical_missing_data(df, col, missing_data_cols)
    elif col in numerical_cols:
        df[col] = impute_continuous_missing_data(df, col, missing_data_cols)
    else:
        pass


df.isnull().sum().sort_values(ascending=False)


print("_________________________________________________________________________________________________________________________________________________")

sns.set(rc={"axes.facecolor":"#87CEEB","figure.facecolor":"#EEE8AA"}) 

palette = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])

plt.figure(figsize=(10,8))

# FIXED THE FOR LOOP BELOW

for i, col in enumerate(col):
    plt.subplot(3,2,i+1)
    sns.boxenplot(color=palette[i % len(palette)])  
    plt.title(i)

plt.show()
##E6E6FA


df[df['trestbps']==0]

df= df[df['trestbps']!=0]

sns.set(rc={"axes.facecolor":"#B76E79","figure.facecolor":"#C0C0C0"})
modified_palette = ["#C44D53", "#B76E79", "#DDA4A5", "#B3BCC4", "#A2867E", "#F3AB60"]
cmap = ListedColormap(modified_palette)

plt.figure(figsize=(10,8))

# FIXED THE FOR LOOP BELOW

for i, col in enumerate(col):
    plt.subplot(3,2,i+1)
    sns.boxenplot( color=palette[i % len(palette)])  
    plt.title(col)

plt.show()

df.trestbps.describe()

df.describe()

print("___________________________________________________________________________________________________________________________________________________________________")


sns.set(rc={"axes.facecolor": "#FFF9ED", "figure.facecolor": "#FFF9ED"})


night_vision_palette = ["#00FF00", "#FF00FF", "#00FFFF", "#FFFF00", "#FF0000", "#0000FF"]

# FIXED THE FOR LOOP BELOW

plt.figure(figsize=(10, 8))
for i, col in enumerate(col):
    plt.subplot(3,2,i+1)
    sns.boxenplot( color=palette[i % len(palette)]) 
    plt.title(col)

plt.show()

df.age.describe()

palette = ["#999999", "#666666", "#333333"]

sns.histplot(data=df,
             x='trestbps',
             kde=True,
             color=palette[0])

plt.title('Resting Blood Pressure')
plt.xlabel('Pressure (mmHg)')
plt.ylabel('Count')

plt.style.use('default')
plt.rcParams['figure.facecolor'] = palette[1]
plt.rcParams['axes.facecolor'] = palette[2]

sns.histplot(df, x='trestbps', kde=True, palette = "Spectral", hue ='sex')

df.info()
df.columns
df.head()

X = df.drop('num', axis=1)
y = df['num']

# FIXED THE ENCODING CODE BELOW

onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(exclude=['object']).columns

for col in categorical_cols:
    X[col] = X[col].astype(str)

X_encoded = X.copy()  # Copy X to preserve original data

# Apply OneHotEncoder to categorical columns
onehot_encoded = onehot_encoder.fit_transform(X_encoded[categorical_cols])
onehot_encoded_df = pd.DataFrame(onehot_encoded, index=X_encoded.index, columns=onehot_encoder.get_feature_names_out(categorical_cols))

# Combine encoded columns with numeric columns
X_encoded = pd.concat([X_encoded[numeric_cols], onehot_encoded_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.1, random_state=42)

# DELETED THE IMPORTS AS DID ALREADY 

from sklearn.pipeline import Pipeline

# FIXED THE CODE BELOW

models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('KNeighbors Classifier', KNeighborsClassifier()),
    ('Decision Tree Classifier', DecisionTreeClassifier(random_state=42)),
    ('AdaBoost Classifier', AdaBoostClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('XGBoost Classifier', XGBClassifier(random_state=42)),
    ('Support Vector Machine', SVC(random_state=42)),
    ('Naive Bayes Classifier', GaussianNB())
]

best_model = None
best_accuracy = 0.0

# FIXED MANY SMALL SYANTAX ERROS SOME METENTIONED BELOW

for name, model in models:
   
    pipeline = Pipeline([
        ('model', model)  # 'model' instead of 'name'
    ])

    scores = cross_val_score(pipeline, X_train, y_train, cv=5) 

    mean_accuracy = scores.mean() # scores.mean instead of scores.avg

    pipeline.fit(X_train, y_train) 

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Model:", name)
    print("Cross-validation accuracy:", mean_accuracy)
    print("Test Accuracy:", accuracy)
    print()

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline


print("Best Model:", best_model)

# FIXED THE WHOLE CODE BELOW 

categorical_cols = ['thal', 'ca', 'slope', 'exang', 'restecg', 'fbs', 'cp', 'sex']

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def identify_column_types(X):
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    return list(numeric_cols), list(categorical_cols)

numeric_cols, categorical_cols = identify_column_types(X)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

def evaluate_classification_models(X, y):
    
    numeric_cols, categorical_cols = identify_column_types(X)

   
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    models = {
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(),
        "NB": GaussianNB(),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier()
    }

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    best_model = None
    best_accuracy = 0.0
    for name, model in models.items():
      
        if name == "NB":
           
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
                ('classifier', model)
            ])
        else:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        results[name] = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name

    return results, best_model

# Example usage:
results, best_model = evaluate_classification_models(X, y)
print("Model accuracies:", results)
print("Best model:", best_model)

# HYPER-PARAMETER TUNING CODE IS WORKING FINE BUT ITS TAKING A LOT OF TIME TO RUN 

X = df[categorical_cols]  
y = df['num'] 

def hyperparameter_tuning(X, y, categorical_columns, models):
    results = {}

    models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'NB': GaussianNB(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

    X_encoded = pd.get_dummies(X, columns=categorical_columns)

    X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    for model_name, model in models.items():
        param_grid = {}
        if model_name == 'Logistic Regression':
            param_grid = {'C': [0.1, 1, 10, 100]}
        elif model_name == 'KNN':
            param_grid = {'n_neighbors': [3, 5, 7, 9]}
        elif model_name == 'NB':
            param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
        elif model_name == 'SVM':
            param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]}
        elif model_name == 'Decision Tree':
            param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        elif model_name == 'Random Forest':
            param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        elif model_name == 'XGBoost':
            param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
        elif model_name == 'GradientBoosting':
            param_grid = {
                "loss": ["log_loss", "exponential"],
                "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_depth": [3, 5, 8],
                "max_features": ["log2", "sqrt"],
                "criterion": ["friedman_mse", "squared_error"],
                "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
                "n_estimators": [100, 200, 300]
            }
        elif model_name == 'AdaBoost':
            param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200]}

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        results[model_name] = {'best_params': best_params, 'accuracy': accuracy}

    return results

# Example usage:
results = hyperparameter_tuning(X, y, categorical_cols, models)
for model_name, result in results.items():
    print("Model:", model_name)
    print("Best hyperparameters:", result['best_params'])
    print("Accuracy:", result['accuracy'])
    print()
