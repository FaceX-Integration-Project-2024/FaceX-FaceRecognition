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
    frame_counter = 0 
    recognition_interval = 10 
    previous_faces = {} 

    while True:
        success, img = cap.read()
        if not success:
            print("Erreur de lecture de la webcam.")
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgS = cv2.resize(img_rgb, (0, 0), None, 0.25, 0.25)  

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for face_index, (encodeFace, faceLoc) in enumerate(zip(encodesCurFrame, facesCurFrame)):
            top, right, bottom, left = faceLoc
            face_image = img_rgb[top * 4:bottom * 4, left * 4:right * 4] 

            if frame_counter % recognition_interval == 0:
                try:
                    embedding_captured = DeepFace.represent(img_path=face_image, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
                    embedding_captured = normalize(embedding_captured)
                except Exception as e:
                    print(f"Erreur d'extraction de l'embedding avec DeepFace : {e}")
                    continue

                min_distance = float("inf")
                identified_person = None

                print("\n### Comparaison des distances avec la base de données ###")
                for person, embeddings in face_db.items():
                    for embedding in embeddings:
                        embedding = normalize(np.array(embedding))
                        distance = np.linalg.norm(embedding_captured - embedding)

                        if distance < min_distance:
                            min_distance = distance
                            identified_person = person

                seuil_deepface = 0.6  
                if min_distance < seuil_deepface:
                    print(f"\nPersonne identifiée : {identified_person} avec une distance de : {min_distance}")
                    confidence = "certain"  
                else:
                    print("Personne détectée mais avec incertitude")
                    confidence = "incertain"  
                    identified_person = identified_person if identified_person else "Inconnu"

                previous_faces[face_index] = (identified_person, confidence)
            else:
                identified_person, confidence = previous_faces.get(face_index, ("Inconnu", "incertain"))

            if confidence == "certain":
                color = (0, 255, 0)  
            else:
                color = (0, 165, 255)  

            cv2.rectangle(img, (left * 4, top * 4), (right * 4, bottom * 4), color, 2)
            cv2.putText(img, identified_person, (left * 4, top * 4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        frame_counter += 1 

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
