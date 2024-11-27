from config.env_loader import load_env_variables
from database.supabase_client import create_supabase_client
from database.attendance import (
    get_active_class_students_face_data,
    get_attendance_for_block,
    post_student_attendance_db
)
from face_recognition.processing import recognize_faces_in_frame
from face_recognition.validation import check_face_data_validity
import cv2
from datetime import datetime, timedelta

def main():
    env_vars = load_env_variables()
    supabase = create_supabase_client(env_vars['DB_URL'], env_vars['DB_KEY'])

    block_id, face_db = get_active_class_students_face_data(supabase, env_vars['LOCAL'])
    existing_attendance = get_attendance_for_block(supabase, block_id)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la webcam")
        return

    last_block_check_time = datetime.now()
    block_change_check_interval = timedelta(minutes=5)

    while True:
        if datetime.now() - last_block_check_time > block_change_check_interval:
            block_id, face_db = get_active_class_students_face_data(supabase, env_vars['LOCAL'])
            existing_attendance = get_attendance_for_block(supabase, block_id)
            last_block_check_time = datetime.now()

        success, img = cap.read()
        if not success:
            print("Erreur de lecture de la webcam.")
            break

        recognize_faces_in_frame(img, face_db, supabase, block_id, existing_attendance)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == "__main__":
    main()
