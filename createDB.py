import os
import pandas as pd
import json
from deepface import DeepFace

# Dossier contenant les images des visages connus (chaque sous-dossier représente une personne)
known_faces_dir = "known_faces/"  # Exemple de structure : known_faces/John, known_faces/Mary, etc.

# Modèle et backend utilisés pour extraire les embeddings
model_name = "Facenet512"
detector_backend = "retinaface"

# Créer un dictionnaire vide pour stocker les embeddings
face_db = {}

# Parcourir chaque sous-dossier (chaque sous-dossier correspond à une personne)
for person_name in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, person_name)
    
    # Parcourir chaque image dans le sous-dossier de la personne
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        
        # Extraire l'embedding de l'image
        try:
            print(f"Traitement de l'image : {image_path}")
            
            # Extraire les embeddings, retourne une liste même si un seul embedding est trouvé
            embedding_objs = DeepFace.represent(img_path=image_path, model_name=model_name, detector_backend=detector_backend)
            
            # Inspection de l'objet retourné par DeepFace
            print(f"Object retourné par DeepFace.represent : {embedding_objs}")
            print(f"Type : {type(embedding_objs)} - Longueur : {len(embedding_objs)}")

            # Vérifier si des embeddings ont été extraits
            if embedding_objs and len(embedding_objs) > 0:
                # Prendre le premier embedding de la liste
                embedding = embedding_objs[0]["embedding"]
                
                # Ajouter l'embedding dans le dictionnaire sous la clé de la personne
                if person_name not in face_db:
                    face_db[person_name] = []  # Initialiser la clé pour cette personne
                face_db[person_name].append(embedding)
                print(f"Ajouté : {person_name} - {image_name} avec embedding de taille {len(embedding)}")

            else:
                print(f"Aucun embedding extrait pour {image_name}.")
                
        except Exception as e:
            print(f"Erreur lors de l'extraction de l'embedding pour {image_name} : {e}")

# Sauvegarder la base de données dans un fichier JSON structuré par personnes
with open("face_database_structured.json", "w") as json_file:
    json.dump(face_db, json_file, indent=4)
    print("Base de données de visages créée et sauvegardée dans 'face_database_structured.json'")
