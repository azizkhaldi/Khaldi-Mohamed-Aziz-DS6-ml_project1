import xgboost as xgb
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Charger les données (tu peux remplacer cela par tes propres données)
data = load_iris()
X = data.data
y = data.target

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle XGBoost
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Entraîner le modèle
model.fit(X_train, y_train)

# Sauvegarder le modèle entraîné dans un fichier
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modèle XGBoost entraîné et sauvegardé avec succès.")

