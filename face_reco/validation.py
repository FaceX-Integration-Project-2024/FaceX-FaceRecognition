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
        return face_data

    if not np.issubdtype(embedding.dtype, np.number):
        print(f"Face data non numérique pour {person}")
        face_data = update_face_data(supabase, person)
        if face_data:
            face_db[person] = face_data
        return face_data

    if len(embedding) != 128:
        print(f"Face data de longueur incorrecte pour {person}: {len(embedding)}")
        face_data = update_face_data(supabase, person)
        if face_data:
            face_db[person] = face_data
        return face_data

    return True

def normalize(embedding):
    """
    Normalise le vecteur d'embedding pour un visage.
    """
    return embedding / np.linalg.norm(embedding)
