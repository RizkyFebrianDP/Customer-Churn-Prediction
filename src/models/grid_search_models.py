import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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
    ('classifier', RandomForestClassifier())
])

param = [
    {
    'classifier__n_estimators': [50, 100, 150, 200], 
    'classifier__max_depth': [5, 10, 15, None],  
    'classifier__min_samples_split': [2, 5, 10],  
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2'] 
}
]

#GridSearchCv
search_cv = GridSearchCV(
    estimator=pipe,
    param_grid=param,
    cv=5,
    scoring='accuracy'
)

# --- 5. TRAIN MODELS ---
print("Memulai pencarian hyperparameter dan model Terbaik")
search_cv.fit(X_train, y_train)

# --- 6. BEST PARAMETER ---
print("\n====================================")
print("Pencarian Selesai! Hasil Terbaik:")
print(f"Skor Cross-Validation Terbaik: {search_cv.best_score_:.4f}")
print(f"Kombinasi Hyperparameter Terbaik: {search_cv.best_params_}")
print(f"Model Terbaik: {search_cv.best_estimator_}")
print("====================================")

# --- 7. EVALUATE MODELS ---
best_model = search_cv.best_estimator_
y_pred = best_model.predict(X_test)
accuracy_random_search = accuracy_score(y_test, y_pred)
print(f"Accuracy Random Search: {accuracy_random_search:.4f}")
print("\nLaporan Klasifikasi Model Terbaik pada Data Testing:")
print(classification_report(y_test, y_pred))
print("\n")
print("=================================================")


