import cv2
import json
import numpy as np
import face_recognition
from deepface import DeepFace

with open('face_database_structured.json', 'r') as f:
    face_db = json.load(f)

def normalize(embedding):
    """Normalise un vecteur d'embedding pour qu'il ait une norme de 1."""
    return embedding / np.linalg.norm(embedding)

cap = cv2.VideoCapture(0)
print("Webcam démarrée")

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam")
else:
    while True:
        succes, img = cap.read()
        if not succes:
            print("Erreur de lecture de la webcam.")
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imgS = cv2.resize(img_rgb, (0, 0), None, 0.25, 0.25)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            top, right, bottom, left = faceLoc
            face_image = img_rgb[top*4:bottom*4, left*4:right*4]  

            try:
                embedding_captured = DeepFace.represent(img_path=face_image, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
                embedding_captured = normalize(embedding_captured)
            except Exception as e:
                print(f"Erreur d'extraction de l'embedding avec DeepFace : {e}")
                continue

            identified_person = None
            min_distance = float("inf")

            print("\n### Comparaison des distances avec la base de données ###")
            for person, embeddings in face_db.items():
                for embedding in embeddings:
                    embedding = normalize(np.array(embedding))  
                    distance = np.linalg.norm(embedding_captured - embedding)
                    
                    print(f"Distance avec {person} : {distance}")

                    if distance < min_distance:
                        min_distance = distance
                        identified_person = person

            seuil_deepface = 0.6 
            if min_distance < seuil_deepface:
                print(f"\nPersonne identifiée : {identified_person} avec une distance de : {min_distance}")
                cv2.rectangle(img, (left * 4, top * 4), (right * 4, bottom * 4), (0, 255, 0), 2)
                cv2.putText(img, identified_person, (left * 4, top * 4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print("Personne détectée mais non identifiée")
                cv2.rectangle(img, (left * 4, top * 4), (right * 4, bottom * 4), (0, 0, 255), 2)

        cv2.imshow('Webcam', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
