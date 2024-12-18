import cv2
import numpy as np
import face_recognition
from database.attendance import postStudentAttendanceDB
from utilitaire.lcd import lcd_init, lcd_set_cursor, lcd_write
import RPi.GPIO as GPIO


from PIL import Image
import io

def lcd(person):
    
    lcd_init()
    print("LCD initialisé.")

    # # Exemple d'affichage
    # lcd_set_cursor(0, 0)  # Ligne 1, colonne 0
    # lcd_write("Hello FaceX!")
    # print("Message affiché : Hello FaceX!")

    # lcd_set_cursor(1, 0)  # Ligne 2, colonne 0
    lcd_write(person)
    print("Message affiché : personne")
    
    # Nettoyer les broches GPIO en quittant
    GPIO.cleanup()
    print("GPIO nettoyées.")

def normalize(embedding):
    return embedding / np.linalg.norm(embedding)

def recognize_faces(img, face_db, existing_attendance, supabase, block_id):
    """
    retrouver les visage dans l'image. 
    - S'il trouve personne : renvois False
    - S'il trouve un visage pas dans existing_attendance : il le met présent dans la DB & retourne True
    - S'il trouve un visage déja dans existing_attendance : il retourne None
    
    """

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (0, 0), None, 0.5, 0.5)

    faces_cur_frame = face_recognition.face_locations(img_resized)
    encodes_cur_frame = face_recognition.face_encodings(img_resized, faces_cur_frame)
    for encode_face in encodes_cur_frame:
        min_distance, identified_person = float("inf"), None

        for person, embeddings in face_db.items():
            for embedding in embeddings:
                distance = np.linalg.norm(encode_face - normalize(np.array(embedding)))
                if distance < min_distance:
                    min_distance, identified_person = distance, person

        if identified_person:
            if identified_person not in existing_attendance:
                print(f"Visage reconnu : {identified_person} avec une distance de {min_distance:.2f}")
                postStudentAttendanceDB(supabase, identified_person, block_id)
                existing_attendance.add(identified_person)
                return True
            else:
                print(f"{identified_person} déjà enregistré avec une distance de {min_distance:.2f}")
                lcd(str(identified_person))
                print("reconnu")
                return None
        else:
            print("Visage détecté mais non reconnu.")
            return False

def studentsImgToFaceData(supabase, email):
    """
    Récupère les données faciales d'un étudiant à partir de son email.
    Télécharge l'image depuis Supabase et extrait l'encodage facial.
    """
    try:
        # Récupérer l'image binaire depuis Supabase
        response = supabase.rpc('get_user_by_email', {"user_email": email}).execute()

        if not response.data:  # Si aucun utilisateur n'est trouvé
            print(f"Aucun utilisateur trouvé pour l'email {email}.")
            return None  # Retourne None si l'email n'existe pas
        
        matricule = response.data['matricule']

        binary_data = supabase.storage.from_("id-pictures").download(f'students/{matricule}.jpg')

        # Créer une image à partir des données binaires
        image = Image.open(io.BytesIO(binary_data))

        # Convertir en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Extraire les encodages faciaux
        img_array = np.array(image)
        face_encodings = face_recognition.face_encodings(img_array)

        if face_encodings:
            return face_encodings[0].tolist()
        else:
            print(f"Aucun visage détecté pour {email}.")
            return None
    except Exception as e:
        print(f"Erreur lors du traitement de l'image pour {email}: {e}")