from utils.image_processing import img_to_face_data

def update_face_data(supabase, email):
    """
    Met à jour les données faciales d'un utilisateur en fonction de son email.
    """
    face_data = [img_to_face_data(supabase, email)]
    supabase.rpc('update_face_data', {
        "user_email": email,
        "new_face_data": face_data
    }).execute()
    return face_data
