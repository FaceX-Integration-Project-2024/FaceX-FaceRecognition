import numpy as np
from face_recognition.validation import normalize
import face_recognition
def recognize_faces_in_frame(img, face_db, supabase, block_id, existing_attendance):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (0, 0), None, 0.5, 0.5)

    face_locations = face_recognition.face_locations(img_resized)
    face_encodings = face_recognition.face_encodings(img_resized, face_locations)

    for encode_face in face_encodings:
        identified_person, min_distance = find_closest_match(encode_face, face_db)
        if identified_person not in existing_attendance:
            post_student_attendance_db(supabase, identified_person, block_id)
            existing_attendance.add(identified_person)

def find_closest_match(encode_face, face_db):
    min_distance = float("inf")
    identified_person = None

    for person, embeddings in face_db.items():
        for embedding in embeddings:
            embedding = normalize(np.array(embedding))
            distance = np.linalg.norm(encode_face - embedding)
            if distance < min_distance:
                min_distance = distance
                identified_person = person

    return identified_person, min_distance
