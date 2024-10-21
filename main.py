import os
import cv2
import numpy as np
import face_recognition
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

# Fonction pour envoyer la présence à la DB
def postStudentAttendanceDB(student_email: str, block_id: int, timestamp: str = None, status: str = 'Present'):
    """
    Envoie la présence d'un étudiant pour un bloc spécifique à la base de données.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    
    try:
        # Convertir le block_id en int pour compatibilité avec bigint
        response = supabase.rpc('post_new_attendance', {
            "attendance_student_email": student_email,
            "attendance_block_id": int(block_id),  # conversion ici
            "attendance_status": status,
            "attendance_timestamp": timestamp
        }).execute()

        if response.data:
            print(f"L'étudiant {student_email} a bien été enregistré sur le block_ID N°{block_id}")
        else:
            print(f"L'étudiant {student_email} a déjà été enregistré sur le block_ID N°{block_id}")

    except Exception as e:
        print(f"Erreur lors de l'envoi des présences dans la DB : {e}")




# Fonction pour récupérer les données des étudiants pour une classe donnée
def getActiveClassStudentsFaceData(local):
    """
    Récupère les données (face_data) des étudiants qui ont cours dans la classe donnée
    """
    try:
        response = supabase.rpc("get_active_class_students_face_data", {"local_now": local}).execute()
        return response.data["block_id"], response.data["students"]
    
    except Exception as e:
        print(f"Erreur dans la récupération des données (face_data) depuis la DB : {e}")

def getAttendanceForBlock(class_block_id):
    """
    Récupère les présences déjà enregistrées pour un class_block_id donné.
    """
    try:
        # Convertir le block_id en int pour compatibilité
        response = supabase.rpc("get_attendance_for_class_block", {"class_block_id": int(class_block_id)}).execute()
        if response.data:
            # S'assurer que les identifiants sont bien convertis
            return {attendance['student_email'] for attendance in response.data}
        else:
            return set()
    except Exception as e:
        print(f"Erreur lors de la récupération des présences pour le class_block_id {class_block_id} : {e}")
        return set()




# Fonction pour vérifier le changement de bloc
def checkForBlockChange(current_block_id, local):
    """
    Vérifie si le bloc de cours a changé.
    """
    new_block_id, _ = getActiveClassStudentsFaceData(local)
    if new_block_id != current_block_id:
        print(f"Le bloc de cours a changé de {current_block_id} à {new_block_id}.")
        return new_block_id
    return current_block_id


# Fonction pour rafraîchir les présences à intervalle régulier
def refreshAttendanceForBlock(block_id, refresh_interval_minutes=5):
    """
    Rafraîchit les présences pour un block_id donné toutes les 'refresh_interval_minutes'.
    """
    now = datetime.now()
    if now - refreshAttendanceForBlock.last_refresh_time > timedelta(minutes=refresh_interval_minutes):
        refreshAttendanceForBlock.last_refresh_time = now
        return getAttendanceForBlock(block_id)  # Requête à la base de données pour récupérer les présences mises à jour
    return None

# Initialiser la dernière vérification de rafraîchissement
refreshAttendanceForBlock.last_refresh_time = datetime.now() - timedelta(minutes=6)  # Forcer le premier rafraîchissement immédiatement


def normalize(embedding):
    return embedding / np.linalg.norm(embedding)


# Charger les variables d'environnement
load_dotenv()
DB_URL = os.getenv('DB_URL')
DB_KEY = os.getenv('DB_KEY')
LOCAL = os.getenv('LOCAL')

# Initialiser la connexion DB
supabase: Client = create_client(DB_URL, DB_KEY)

# Récupérer les infos des étudiants et block_id depuis la DB
block_id, face_db = getActiveClassStudentsFaceData(LOCAL)
print(f"Vous êtes dans le local : {LOCAL} avec un block_id : {block_id}")

# Récupérer les présences existantes pour le class_block_id
existing_attendance = getAttendanceForBlock(block_id)


cap = cv2.VideoCapture(0)
print("Webcam démarrée")

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam")
else:
    frame_counter = 0
    recognition_interval = 10
    previous_faces = {}
    face_ids = {}
    detection_timestamps = {}
    block_change_check_interval = timedelta(minutes=5)  # Vérification du changement de bloc toutes les 5 minutes
    last_block_check_time = datetime.now()

    while True:
        # Vérifier si le bloc de cours a changé toutes les 5 minutes
        if datetime.now() - last_block_check_time > block_change_check_interval:
            block_id = checkForBlockChange(block_id, LOCAL)
            existing_attendance = getAttendanceForBlock(block_id)
            last_block_check_time = datetime.now()

        # Rafraîchir les présences si nécessaire
        updated_attendance = refreshAttendanceForBlock(block_id)
        if updated_attendance is not None:
            existing_attendance = updated_attendance
            print(f"Les présences pour le block {block_id} ont été rafraîchies.")

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

            # Vérifier si la personne a déjà été enregistrée pour ce block
            if identified_person != "Inconnu" and identified_person not in existing_attendance:
                postStudentAttendanceDB(identified_person, block_id)
                existing_attendance.add(identified_person)  # Mettre à jour localement
            else:
                print(f"{identified_person} a déjà été enregistré pour ce bloc.")

        frame_counter += 1

        # Quitter la boucle si 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # cv2.destroyAllWindows()  # Désactivé pour Raspberry Pi
