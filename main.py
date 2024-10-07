import cv2
import json
import numpy as np
import face_recognition

with open('face_database_structured.json', 'r') as f:
    face_db = json.load(f)

def normalize(embedding):
    return embedding / np.linalg.norm(embedding)

cap = cv2.VideoCapture(0)
print("Webcam démarrée")

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam")
else:
    frame_counter = 0
    recognition_interval = 10 
    previous_faces = {}  
    face_ids = {}  

    while True:
        success, img = cap.read()
        if not success:
            print("Erreur de lecture de la webcam.")
            break

        # Convertir l'image en RGB et réduire la taille pour accélérer le traitement
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgS = cv2.resize(img_rgb, (0, 0), None, 0.5, 0.5)

        # Détection des visages et extraction des encodings
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        # Associer les visages actuels aux visages précédents
        current_face_ids = {}

        for face_index, (encodeFace, faceLoc) in enumerate(zip(encodesCurFrame, facesCurFrame)):
            top, right, bottom, left = faceLoc
            top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2  # Ajuster les coordonnées à l'image d'origine

            # Initialiser les variables de distance minimale et d'identification
            min_distance = float("inf")
            identified_person = None
            current_face_id = None

            # Comparaison des encodages actuels avec les encodages de la base de données
            for person, embeddings in face_db.items():
                for embedding in embeddings:
                    embedding = normalize(np.array(embedding))
                    distance = np.linalg.norm(encodeFace - embedding)

                    if distance < min_distance:
                        min_distance = distance
                        identified_person = person

            seuil_facerecognition = 0.6
            if min_distance < seuil_facerecognition:
                print(f"Personne identifiée : {identified_person} avec une distance de : {min_distance}")
                confidence = "certain"
            else:
                print(f"Personne détectée mais avec incertitude avec une distance de : {min_distance}")
                confidence = "incertain"
                identified_person = "Inconnu"

            # Associer ce visage à un identifiant unique (recherche dans les précédents encodages)
            for prev_id, (prev_encoding, prev_loc) in previous_faces.items():
                prev_distance = np.linalg.norm(encodeFace - prev_encoding)
                if prev_distance < 0.5:  # Seuil pour déterminer si le visage correspond à un visage précédent
                    current_face_id = prev_id
                    break

            # Si le visage actuel ne correspond à aucun visage précédent, créer un nouvel ID
            if current_face_id is None:
                current_face_id = len(face_ids) + 1
                face_ids[current_face_id] = identified_person

            # Mise à jour de l'identification actuelle
            current_face_ids[current_face_id] = (encodeFace, faceLoc)
            previous_faces[current_face_id] = (encodeFace, faceLoc)  # Mettre à jour les informations du visage

            # Afficher le cadre et le nom de la personne
            color = (0, 255, 0) if confidence == "certain" else (0, 165, 255)
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            cv2.putText(img, f"{identified_person} (ID: {current_face_id})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Mise à jour des informations précédentes
        previous_faces = current_face_ids

        frame_counter += 1  # Incrémenter le compteur de frames

        # Afficher l'image de la webcam
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libération des ressources
    cap.release()
    cv2.destroyAllWindows()
