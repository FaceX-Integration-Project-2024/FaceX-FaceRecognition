from utilitaire.face_recognition_utils import studentsImgToFaceData

def update_face_data(supabase, email):
    face_data = [studentsImgToFaceData(supabase, email)]
    
    try:
        response = supabase.rpc('update_face_data', {
            "user_email": email,
            "new_face_data": face_data
        }).execute()

        # Access the error and status correctly
        if response.get("error"):  # Use .get() to safely check for "error" key
            raise Exception(f"Erreur lors de la mise à jour des données faciales : {response['error']}")

        return face_data

    except Exception as e:
        print(f"Erreur lors de l'exécution de la procédure stockée pour {email}: {e}")
        return None
