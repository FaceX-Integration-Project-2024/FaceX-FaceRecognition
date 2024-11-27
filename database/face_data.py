from utilitaire.face_recognition_utils import studentsImgToFaceData

def update_face_data(supabase, email):
    face_data = [studentsImgToFaceData(supabase, email)]
    supabase.rpc('update_face_data', {
        "user_email": email,
        "new_face_data": face_data
    }).execute()
    return face_data
