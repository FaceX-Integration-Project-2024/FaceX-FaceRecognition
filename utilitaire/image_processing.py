import face_recognition
import numpy as np

def img_to_face_data(supabase, email):
    binary_data = supabase.storage.from_("id-pictures").download(f'students/{email}.jpg')
    img_array = np.array(Image.open(io.BytesIO(binary_data)))
    face_encodings = face_recognition.face_encodings(img_array)
    return face_encodings[0].tolist() if face_encodings else None
