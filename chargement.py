import pickle

# Charger le modèle avec pickle
with open("iaq.pkl", 'rb') as f:
    model = pickle.load(f)
