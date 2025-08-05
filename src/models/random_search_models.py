# --- 1. IMPORT LIBRARY ---
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)

# --- 2. DATASET INFORMATION ---
dataset = pd.read_csv("Telco-Customer-Churn.csv")
print("Informasi Dataset")
print("====================================================")
print(dataset.tail())
print("\n")
print(dataset.info())
print("\n")
print(dataset.describe())
print("\n")
print(dataset.isna().sum())
print("\n")
print(dataset.duplicated().sum())
print("====================================================")

# --- 3. PREPROCESSING ---

#Check Distirbusi Variabel Y
print('Total Distribusi :',dataset['Churn'].value_counts())

# Menghapus kolom yang tidak bernilai
dataset.drop(['customerID'], axis=1, inplace=True)

# Mengatasi masalah pada kolom 'TotalCharges'
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
dataset['TotalCharges'].fillna(dataset['TotalCharges'].median(), inplace=True)

# Memisahkan fitur numerik dan kategorikal
categorical_features = dataset.select_dtypes(include=['object']).columns.drop('Churn')
numerical_features = dataset.select_dtypes(include=np.number).columns

# Menerapkan One-Hot Encoding pada fitur kategorikal
dataset = pd.get_dummies(dataset, columns=categorical_features, drop_first=True)

# Mengubah Nilai Chrun
dataset['Churn'] = dataset['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

#Split Dataset
X = dataset.drop('Churn', axis=1)
y = dataset['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- 4. SEARCH BEST MODEL WITH RANDOMIZED SEARCH CV ---
pipe = ImbPipeline([
    ('SMOTE', SMOTE(random_state=42)),
    ('Scaler', StandardScaler()),
    ('classifier', [CatBoostClassifier()])
])

param = [
    #Logisitik Regresion Parameter
    {
        'classifier' : [LogisticRegression(max_iter=150,random_state=42)],
        'classifier__penalty' : ['l1','l2'],
        'classifier__C' : [0.1,1,10],
        'classifier__solver' : ['liblinear', 'saga']

    },
    #Gradient Boosting Parameter
    {
        'classifier' : [GradientBoostingClassifier(random_state=42)],
        'classifier__learning_rate' : [0.1,0.5,1],
        'classifier__n_estimators' : [10,50,100],
        'classifier__max_depth' : [None,3,5,7],
    },
    #Random Forest Parameter
    {
        'classifier' : [RandomForestClassifier(random_state=42)],
        'classifier__n_estimators' : [10,50,100],
        'classifier__max_depth' : [None,3,5,7],
        'classifier__min_samples_split' : [2,5,10],
        'classifier__min_samples_leaf' : [1,2,4],
    },
    #SVC Parameter
    {
        'classifier' : [SVC(random_state=42)],
        'classifier__kernel' : ['linear','rbf','poly'],
        'classifier__C' : [0.1,1,10],
        'classifier__degree' : [2,3,4],
    },
    #LGBM Parameter
    {
        'classifier' : [LGBMClassifier(random_state=42)],
        'classifier__n_estimators' : [10,50,100],
        'classifier__learning_rate' : [0.1,0.5,1],
        'classifier__max_depth' : [3,5,7],
        'classifier__class_weight' : ['balanced', None]
    },
    
    #CatBoost Parameter
    {
        'classifier' : [CatBoostClassifier(random_state=42)],
        'classifier__n_estimators' : [10,50,100],
        'classifier__learning_rate' : [0.1,0.5,1],
        'classifier__max_depth' : [3,5,7]
    }
]

#RandomSearchCV
random_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param,
    n_iter=50,
    cv=5,
    scoring='accuracy'
)

# --- 5. TRAIN MODELS ---
print("Memulai pencarian hyperparameter dan model Terbaik")
random_search.fit(X_train, y_train)

# --- 6. BEST PARAMETER ---
print("\n====================================")
print("Pencarian Selesai! Hasil Terbaik:")
print(f"Skor Cross-Validation Terbaik: {random_search.best_score_:.4f}")
print(f"Kombinasi Hyperparameter Terbaik: {random_search.best_params_}")
print(f"Model Terbaik: {random_search.best_estimator_}")
print("====================================")

# --- 7. EVALUATE MODELS ---
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy_random_search = accuracy_score(y_test, y_pred)
print(f"Accuracy Random Search: {accuracy_random_search:.4f}")
print("\nLaporan Klasifikasi Model Terbaik pada Data Testing:")
print(classification_report(y_test, y_pred))
print("\n")
print("=================================================")
