import cv2
import json
import numpy as np
import face_recognition

#Je charge la DB
with open('face_database_structured.json', 'r') as f:
    face_db = json.load(f)

#Normalise l'embedding
def normalize(embedding):
    return embedding / np.linalg.norm(embedding)

# 1 pour ext 0 pour in
cap = cv2.VideoCapture(1)
print("Webcam démarrée")

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam")
else:
    frame_counter = 0
    recognition_interval = 10  #intervale de 10 frames
    previous_faces = {}  # Stockage des visages identifer précédemment 

    while True:
        success, img = cap.read()
        if not success:
            print("Erreur de lecture de la webcam.")
            break

        # Convertir l'image en RGB & réduire la taille pour accélérer le traitement
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgS = cv2.resize(img_rgb, (0, 0), None, 0.5, 0.5)

        # Détection des visages et extraction des encodings
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for face_index, (encodeFace, faceLoc) in enumerate(zip(encodesCurFrame, facesCurFrame)):
            top, right, bottom, left = faceLoc

            # Recalculer les coordonnées pour l'image d'origine
            top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2

            # Compteurs d'Intervalle de frames pour la reconnaissance
            if frame_counter % recognition_interval == 0:
                min_distance = float("inf")
                identified_person = None

                # Comparaison avec la base de donnée
                print("\n### Comparaison des distances avec la base de données ###")
                for person, embeddings in face_db.items():
                    for embedding in embeddings:
                        embedding = normalize(np.array(embedding))
                        distance = np.linalg.norm(encodeFace - embedding)

                        if distance < min_distance:
                            min_distance = distance
                            identified_person = person

                # Seuil de distance pour la reconnaissance
                seuil_facerecognition = 0.6
                if min_distance < seuil_facerecognition:
                    print(f"Personne identifiée : {identified_person} avec une distance de : {min_distance}")
                    confidence = "certain"
                else:
                    print(f"Personne identifiée mais avec incertitude : {identified_person} avec une distance de : {min_distance}")
                    confidence = "incertain"
                    identified_person = identified_person if identified_person else "Inconnu"

                # Mise à jour des informations précédentes
                previous_faces[face_index] = (identified_person, confidence)
            else:
                identified_person, confidence = previous_faces.get(face_index, ("Inconnu", "incertain"))

            # Affichage du cadre et du nom sur l'image
            if confidence == "certain":
                color = (0, 255, 0)  # Vert pour une identification certaine
            else:
                color = (0, 165, 255)  # Orange pour une identification incertaine

            # Dessiner le rectangle autour du visage
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            cv2.putText(img, identified_person, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        frame_counter += 1  # Incrémenter le compteur de frames

        # Afficher l'image de la webcam
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libération des ressources
    cap.release()
    cv2.destroyAllWindows()
