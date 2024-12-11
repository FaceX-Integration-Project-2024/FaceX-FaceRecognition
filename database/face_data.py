from utilitaire.face_recognition_utils import studentsImgToFaceData


def update_face_data(supabase, email):
    face_data = [studentsImgToFaceData(supabase, email)]
    
    try:
        response = supabase.rpc('update_face_data', {
            "user_email": email,
            "new_face_data": face_data
        }).execute()

        if response.error:
            raise TimeoutError(f"Erreur lors de la mise à jour des données faciales : {response.error}")
        
        return face_data
    except TimeoutError  as e:
        print(f"Erreur lors de l'exécution de la procédure stockée pour {email}: {e}")
