import json
import numpy as np
from scipy.spatial.distance import cosine
from deepface import DeepFace

# Charger la base de données de visages connus depuis le fichier JSON
with open('face_database_structured.json', 'r') as f:
    face_db = json.load(f)

# Chemin de l'image à identifier (tu peux remplacer par le chemin correct de ton image)
img_to_identify = "images/aloisGrimace.jpg"  # Remplace par le chemin de l'image à identifier

# Paramètres du modèle
model_name = "Facenet512"
detector_backend = "retinaface"

# Extraire l'embedding du visage à identifier
try:
    embedding_obj = DeepFace.represent(img_path=img_to_identify, model_name=model_name, detector_backend=detector_backend)[0]
    embedding_to_identify = embedding_obj["embedding"]
except Exception as e:
    print(f"Erreur lors de l'extraction de l'embedding pour l'image à identifier : {e}")
    embedding_to_identify = None

# Si l'embedding a été extrait avec succès
if embedding_to_identify is not None:
    # Trouver la correspondance la plus proche dans la base de données
    min_distance = float("inf")
    identified_person = None

    # Parcourir chaque personne dans la base de données pour comparer les embeddings
    for person, embeddings in face_db.items():
        # Comparer avec chaque embedding de cette personne
        for embedding in embeddings:
            distance = cosine(embedding_to_identify, embedding)
            
            # Mettre à jour la personne identifiée si la distance est plus faible
            if distance < min_distance:
                min_distance = distance
                identified_person = person

    # Afficher la personne identifiée même si la distance est supérieure au seuil
    print(f"Personne identifiée la plus proche : {identified_person} avec une distance de {min_distance:.4f}")

    # Indiquer si cette correspondance est suffisante ou non
    seuil = 0.3  # Ajuste ce seuil selon les besoins de précision
    if min_distance < seuil:
        print(f"La personne est bien identifiée comme {identified_person}.")
    else:
        print(f"Personne identifiée (mais la distance est au-dessus du seuil de {seuil}): {identified_person} avec une distance de {min_distance:.4f}.")
