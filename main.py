import os
import cv2
import numpy as np
import face_recognition
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime, timezone



def postStudentAttendanceDB(student_email: str ,block_id: int, timestamp: str = None , status: str ='Present'):
    """
    Envoie la présence d'un étudiant pour un bloc spécifique à la base de données.
    """

    # Met heure actuelle si aucune heure est spécifier
    if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
    
    try:
        # Envois la présence de l'étudiant dans DB 
        response = supabase.rpc('post_new_attendance', {
            "attendance_student_email": student_email,
            "attendance_block_id": block_id,
            "attendance_status": status,
            "attendance_timestamp": timestamp
        }).execute()

        # véfirie le retours de la DB (true/false)
        if response.data:
            print(f"l'étudants {student_email} à bien été enregistré sur le block_ID N°{block_id}")
        else:
            print(f"L'étudiant {student_email} à déja été enregistré sur le block_ID N°{block_id}")

    except Exception as e:
        print(f"Erreur dans l'envois des Attendence dans la DB: {e}")


def getActiveClassStudentsFaceData(local) :
    """
    Récupere les données (face_data) des étudiants qui ont cours dans la classe donnée 
    """ 

    # récuperer les étudiants qui ont actuellement classe
    try :
        reponse = supabase.rpc("get_active_class_students_face_data", {"local_now":local}).execute()
        return reponse.data["block_id"], reponse.data["students"]
    
    except Exception as e:
        print(f"Erreur dans la récupération des données (face_data) depuis la DB : {e}")


def normalize(embedding):
    return embedding / np.linalg.norm(embedding)





# Récuperer les variables d'environement
load_dotenv()
DB_URL = os.getenv('DB_URL')
DB_KEY = os.getenv('DB_KEY')
LOCAL = os.getenv('LOCAL')


# Initialise conneciton DB
supabase: Client = create_client(DB_URL, DB_KEY)


# Get étudiants info + block_id depuis DB
block_id, face_db = getActiveClassStudentsFaceData(LOCAL)
print(f"Vous etes dans le local : {LOCAL} avec un course_id : {block_id}")


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

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgS = cv2.resize(img_rgb, (0, 0), None, 0.5, 0.5)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        current_face_ids = {}

        for face_index, (encodeFace, faceLoc) in enumerate(zip(encodesCurFrame, facesCurFrame)):
            top, right, bottom, left = faceLoc
            top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2  

            min_distance = float("inf")
            identified_person = None
            current_face_id = None

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

            for prev_id, (prev_encoding, prev_loc) in previous_faces.items():
                prev_distance = np.linalg.norm(encodeFace - prev_encoding)
                if prev_distance < 0.5: 
                    current_face_id = prev_id
                    break

            if current_face_id is None:
                current_face_id = len(face_ids) + 1
                face_ids[current_face_id] = identified_person

            current_face_ids[current_face_id] = (encodeFace, faceLoc)
            previous_faces[current_face_id] = (encodeFace, faceLoc)  # Mettre à jour les informations du visage

            print(f"ID: {current_face_id}, Personne: {identified_person}, Confiance: {confidence}")
            
            # envois à la DB 
            if identified_person != "Inconnu" :
                postStudentAttendanceDB(identified_person, block_id)

            # ### Code pour l'affichage graphique - Désactivé pour Raspberry Pi ###
            # color = (0, 255, 0) if confidence == "certain" else (0, 165, 255)
            # cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            # cv2.putText(img, f"{identified_person} (ID: {current_face_id})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        previous_faces = current_face_ids

        frame_counter += 1  #

        # ### Affichage de l'image désactivé pour Raspberry Pi ###
        # cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # cv2.destroyAllWindows()  # Désactivé pour Raspberry Pi