import pickle

# Charger le modèle avec pickle
with open("model.pkl", 'rb') as f:
    model = pickle.load(f)
