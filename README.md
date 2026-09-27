# 🧠 Détection précoce de la dépression par le machine learning

Application qui estime le risque de dépression chez un étudiant à partir de quelques données comportementales (pression académique, stress financier, heures de travail, habitudes alimentaires…). Le projet couvre toute la chaîne data science : nettoyage des données, sélection de variables, comparaison de modèles et déploiement dans une application web.

> Projet réalisé dans le cadre de ma formation d'ingénieure IA & Data à l'ESME Sudria.

![Aperçu de l'application](assets/demo.png)

## 🎯 Démarche

1. **Nettoyage des données** : suppression des doublons, traitement des valeurs manquantes (médiane pour les variables numériques, modalité la plus fréquente pour les catégorielles) et encodage des variables catégorielles, sur près de 28 000 observations.
2. **Sélection de variables** : filtrage par corrélation avec la cible, puis par importance des variables d'un Random Forest. Six variables sont retenues : pensées suicidaires, âge, pression académique, heures de travail ou d'études, stress financier et habitudes alimentaires.
3. **Modélisation** : comparaison de six modèles (régression logistique, arbre de décision, Random Forest, KNN, XGBoost et Voting Classifier), avec optimisation des hyperparamètres par validation croisée (GridSearchCV).
4. **Évaluation** : rapports de classification, matrices de confusion et courbes ROC/AUC.
5. **Déploiement** : le modèle retenu, la régression logistique, est intégré dans une application Streamlit.

## 📊 Résultats

La régression logistique a été retenue pour ses bonnes performances et son interprétabilité. Sur le jeu de test (30 % des données) :

| Accuracy | Rappel (classe dépression) | F1-score | AUC |
|:---:|:---:|:---:|:---:|
| 84 % | 88 % | 0,87 | 0,91 |

Le rappel est privilégié : dans un contexte de détection précoce, il vaut mieux signaler un cas à tort que passer à côté d'une personne en difficulté.

## 🛠️ Technologies

Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn · Streamlit

## 📁 Structure du projet

```
├── app.py          # Application web Streamlit
├── train.py        # Nettoyage, sélection de variables, entraînement et évaluation
├── models/         # Modèle entraîné et scaler (.pkl)
├── data/           # Emplacement du jeu de données (voir data/README.md)
└── requirements.txt
```

## 🚀 Installation et lancement

```bash
git clone https://github.com/camille-bre-esme/detection-depression-ml.git
cd detection-depression-ml
pip install -r requirements.txt
```

**Lancer l'application** (le modèle déjà entraîné est fourni) :

```bash
streamlit run app.py
```

**Réentraîner les modèles** : télécharger le jeu de données (voir [data/README.md](data/README.md)), puis :

```bash
python train.py
```

## 📚 Données

[Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) (Kaggle) : environ 28 000 réponses d'étudiants décrivant leur profil, leur rythme de vie et leur état de santé mentale.

## ⚠️ Avertissement

Ce projet est un exercice pédagogique de machine learning. Il ne constitue en aucun cas un outil de diagnostic et ne remplace pas l'avis d'un professionnel de santé. En France, le **3114** (numéro national de prévention du suicide) est joignable 24h/24, gratuitement.

## 💡 Pistes d'amélioration

- Effectuer l'imputation et la standardisation après la séparation train/test, dans un `Pipeline` scikit-learn, pour éviter toute fuite de données
- Utiliser un encodage one-hot pour les variables catégorielles sans ordre naturel
- Expliquer chaque prédiction avec SHAP pour rendre le modèle plus transparent
- Déployer l'application en ligne sur Streamlit Community Cloud
