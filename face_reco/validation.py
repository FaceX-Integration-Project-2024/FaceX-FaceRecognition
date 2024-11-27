import numpy as np
from database.face_data import update_face_data

def check_face_data_validity(supabase, person, embedding, face_db):
    """
    Vérifie que les données de visage sont valides et les met à jour si nécessaire.
    """
    if embedding is None or len(embedding) == 0:
        print(f"Face data manquante pour {person}")
        face_data = update_face_data(supabase, person)
        if face_data:
            face_db[person] = face_data
        return face_data

    try:
        embedding = np.array(embedding)
    except:
        print(f"Face data invalide pour {person}")
        face_data = update_face_data(supabase, person)
        if face_data:
            face_db[person] = face_data
        return face_
