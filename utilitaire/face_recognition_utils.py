import cv2
import numpy as np
import face_recognition
# from utilitaire.face_data_utils import normalize # pas importer sinon dépendances en boucle
from database.attendance import postStudentAttendanceDB

from PIL import Image
import io
def recognize_faces(img, face_db, existing_attendance, supabase, block_id):
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

        if identified_person and identified_person not in existing_attendance:
            postStudentAttendanceDB(supabase, identified_person, block_id)
            existing_attendance.add(identified_person)

def studentsImgToFaceData(supabase, email):
    """
    Récupère les données faciales d'un étudiant à partir de son email.
    Télécharge l'image depuis Supabase et extrait l'encodage facial.
    """
    try:
        # Récupérer l'image binaire depuis Supabase
        response = supabase.rpc('get_user_by_email', {"user_email": email}).execute()
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
        return None