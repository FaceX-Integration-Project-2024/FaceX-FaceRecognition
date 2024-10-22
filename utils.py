import os
import json
import face_recognition
from supabase import create_client, Client
from dotenv import load_dotenv
from PIL import Image
import io
import numpy as np

def imgToFaceData(img):
    try:
        print(f"Traitement de l'image.")
        # Vérifier que l'image est bien en format RGB (important)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Traiter l'image en numpy
        img_array = np.array(img)
        
        # Retrouver les visages
        face_encodings = face_recognition.face_encodings(img_array)
        
        # Vérifier si un encodage a été extrait
        if len(face_encodings) > 0:
            embedding = face_encodings[0]
            return embedding.tolist()
        else:
            print(f"Aucun encodage extrait pour l'image.")
            return None
            
    except Exception as e:
        print(f"Erreur lors de l'extraction des encodages : {e}")
        return None


def studentsImgToFaceData(supabase: Client, studentsEmail: str):
    """
    Recoit email d'un étudants et renvoir les face_data de son image stocké dans supabase
    """

    

    # récuperer matricutle étudiant
    response = supabase.rpc('get_user_by_email', {
            "user_email": studentsEmail,
        }).execute()
    matricule = response.data['matricule']
    
    # Télécharger l'image binaire depuis le stockage Supabase
    binary_data = supabase.storage.from_("id-pictures").download('students/' +  matricule + '.jpg')

    # Créer une image à partir des données binaires
    image = Image.open(io.BytesIO(binary_data))
    # image.show()

    # Extraire les données du visage
    encodings = imgToFaceData(image)
    if encodings:
        return encodings
    else:
        print("Aucun encodage trouvé pour cet étudiant.")


# Chargement des variables d'environnement
load_dotenv()
DB_URL = os.getenv('DB_URL')
DB_KEY = os.getenv('DB_KEY')
LOCAL = os.getenv('LOCAL')

supabase: Client = create_client(DB_URL, DB_KEY)


print(studentsImgToFaceData(supabase,'gaetan.carbonnelle1@gmail.com'))
