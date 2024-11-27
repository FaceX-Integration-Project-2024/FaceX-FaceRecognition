import numpy as np
from database.face_data import update_face_data

def checkFaceDataValidity(supabase, person, embedding, face_db):
    if embedding is None or len(embedding) == 0:
        face_db[person] = update_face_data(supabase, person)
        return face_db[person]
    
    embedding = np.array(embedding)
    if not np.issubdtype(embedding.dtype, np.number) or len(embedding) != 128:
        face_db[person] = update_face_data(supabase, person)
        return face_db[person]

    return True

def normalize(embedding):
    return embedding / np.linalg.norm(embedding)
