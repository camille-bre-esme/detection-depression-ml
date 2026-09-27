import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from xgboost import XGBClassifier

# TRAITEMENT DE LA BASE DE DONNEES

df = pd.read_csv("data/depression.csv")

# Vérification des données manquantes
total = df.astype(str).apply(lambda x: x.str.contains(r'^[^a-zA-Z0-9]+$')).sum()
print(total)

print(df.select_dtypes(include=['object', 'int', 'float','str']).isnull().sum())
print(df.dtypes)

# Vérification des données dupliquées
print(df.duplicated().sum())
df = df.drop_duplicates() 

# Suppression des colonnes inutiles
df = df.drop(columns=['id'])

df.dropna(subset=['Sleep Duration', 'Family History of Mental Illness'], inplace=True)

# Remplacer les valeurs manquantes
imputer_num = SimpleImputer(strategy='median')
df[['CGPA', 'Financial Stress']] = imputer_num.fit_transform(df[['CGPA', 'Financial Stress']])

imputer_cat = SimpleImputer(strategy='most_frequent')
df[['Profession', 'Academic Pressure', 'Dietary Habits']] = imputer_cat.fit_transform(df[['Profession', 'Academic Pressure', 'Dietary Habits']])

# Passage de str en int ou float
columns_to_encode = ["Gender", "City", "Profession", "Academic Pressure", "Sleep Duration",
                     "Dietary Habits", "Degree", "Have you ever had suicidal thoughts ?",
                     "Family History of Mental Illness"]
for col in columns_to_encode:
    df[col] = LabelEncoder().fit_transform(df[col])

# ANALYSE DE LA BASE DE DONNEES
X = df.drop('Depression', axis=1)
y = df['Depression']

# Boîte à moustache pour savoir si il faut standardiser ou normaliser
print(np.mean(X))
print(np.std(X))
X.boxplot()
plt.show()

# Corrélation
corr_matrix = df.corr().round(2)
plt.figure(figsize=(12, 10))
sns.heatmap(data=corr_matrix, annot=True)
plt.title("Matrice de corrélation")
plt.show()

matr = df.corr()
corr_with_stage = matr['Depression']
selected_features = corr_with_stage[abs(corr_with_stage) > 0.2].drop('Depression')
X_filtered = X[selected_features.index]

# Feature importances avec Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_filtered, y)

importance_df = pd.DataFrame({
    'Variable': X_filtered.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.barh(X_filtered.columns, rf.feature_importances_)
plt.show()

important_features = importance_df[importance_df['Importance'] > 0.04]['Variable']
X_selected = X_filtered[important_features]

# Standardisation
scaler = StandardScaler()
Xst = scaler.fit_transform(X_selected)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    Xst, y, test_size=0.3, random_state=42, stratify=y)

# ENTRAÎNEMENT DES MODÈLES

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
param_grid_rf = {'n_estimators':[50,75,100,125,150,175,200]}
rf_cv = GridSearchCV(rf_model, param_grid_rf, cv=5)
rf_cv.fit(X_train, y_train)
print("Meilleurs paramètres pour Random Forest:",rf_cv.best_params_)
y_pred_rf = rf_cv.predict(X_test)

# KNN
knn_model = KNeighborsClassifier()
param_grid_knn = {'n_neighbors':np.arange(1,50)}
knn_cv = GridSearchCV(knn_model, param_grid_knn, cv=5)
knn_cv.fit(X_train, y_train)
print("Meilleurs paramètres pour KNN:",knn_cv.best_params_)
y_pred_knn = knn_cv.predict(X_test)

# XGBoost
xgb_model = XGBClassifier()
param_grid_xgb = {
    'max_depth':np.arange(1,10),
    'subsample':np.arange(0.1,1,0.1),
    'n_estimators':[10,20,30,40,50]}
xgb_cv = GridSearchCV(xgb_model, param_grid_xgb, cv=5)
xgb_cv.fit(X_train, y_train)
print("Meilleurs paramètres pour XGBoost:",xgb_cv.best_params_)
y_pred_xgb = xgb_cv.predict(X_test)

# Voting Classifier
voting_model = VotingClassifier(
    estimators=[('LogReg', log_model), ('RandomForest', rf_model), ('XGBoost', xgb_model)], voting='soft')
voting_model.fit(X_train, y_train)
y_pred_voting = voting_model.predict(X_test)

# CLASSIFICATION REPORT DE CHAQUE MODÈLE
models_results = {
    "Logistic Regression": (log_model,    y_pred_log),
    "Decision Tree":       (tree_model,   y_pred_tree),
    "Random Forest":       (rf_cv,     y_pred_rf),
    "KNN":                 (knn_cv,    y_pred_knn),
    "XGBoost":             (xgb_cv,    y_pred_xgb),
    "Voting Classifier":   (voting_model, y_pred_voting),
}

for name, (model, y_pred) in models_results.items():
    print(f"Modèle : {name}")
    print(classification_report(y_test, y_pred))

# MATRICES DE CONFUSION DE CHAQUE MODELE
for model_name, (model, y_pred) in models_results.items():
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot()
    plt.title(f"Matrice de confusion - {model_name}")
    plt.show()

# COURBES ROC + AUC SCORE
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], '--', label="Aléatoire (AUC = 0.5)")

for name, (model, y_pred) in models_results.items():
    y_pred_prob = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
    auc_score = roc_auc_score(y_test, y_pred_prob[:, 1])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

plt.xlabel('Taux faux positif')
plt.ylabel('Taux vrai positif')
plt.title('Courbes ROC et AUC pour la comparaison des modèles')
plt.legend()
plt.show()

import joblib


joblib.dump(log_model, 'models/modele_regression_logistique.pkl')
joblib.dump(scaler, 'models/mon_scaler.pkl')