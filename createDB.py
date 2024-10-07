import os
import json
import face_recognition

# Dossier contenant les images des visages connus (chaque sous-dossier représente une personne)
known_faces_dir = "known_faces/"  # Exemple de structure : known_faces/John, known_faces/Mary, etc.

# Créer un dictionnaire vide pour stocker les embeddings
face_db = {}

# Parcourir chaque sous-dossier (chaque sous-dossier correspond à une personne)
for person_name in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, person_name)
    
    # Parcourir chaque image dans le sous-dossier de la personne
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        
        try:
            print(f"Traitement de l'image : {image_path}")

            # Charger l'image et détecter les visages
            img = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(img)
            
            # Vérifier si un encodage a été extrait
            if len(face_encodings) > 0:
                # Prendre le premier encodage trouvé (si plusieurs visages sont détectés)
                embedding = face_encodings[0]

                # Ajouter l'embedding dans le dictionnaire sous la clé de la personne
                if person_name not in face_db:
                    face_db[person_name] = []  # Initialiser la clé pour cette personne
                face_db[person_name].append(embedding.tolist())  # Convertir en liste pour JSON
                print(f"Ajouté : {person_name} - {image_name} avec embedding de taille {len(embedding)}")
            else:
                print(f"Aucun embedding extrait pour {image_name}.")
                
        except Exception as e:
            print(f"Erreur lors de l'extraction de l'embedding pour {image_name} : {e}")

# Sauvegarder la base de données dans un fichier JSON structuré par personnes
with open("face_database_structured.json", "w") as json_file:
    json.dump(face_db, json_file, indent=4)
    print("Base de données de visages créée et sauvegardée dans 'face_database_structured.json'")
