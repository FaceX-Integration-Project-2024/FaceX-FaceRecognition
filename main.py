import cv2
from datetime import datetime, timedelta
from config.env_loader import load_env_variables
from database.supabase_client import create_supabase_client
from database.attendance import getActiveClassStudentsFaceData, getAttendanceForBlock, postStudentAttendanceDB
from utilitaire.face_data_utils import checkFaceDataValidity, normalize
from utilitaire.face_recognition_utils import recognize_faces

def main():
    env_vars = load_env_variables()
    supabase = create_supabase_client(env_vars['DB_URL'], env_vars['DB_KEY'])

    block_id, face_db = getActiveClassStudentsFaceData(supabase, env_vars['LOCAL'])
    print(f"Vous êtes dans le local : {env_vars['LOCAL']} avec un block_id : {block_id}")

    print("Vérification initiale des face data...")
    for person in face_db:
        checkFaceDataValidity(supabase, person, face_db[person][0], face_db)
    print("Vérification terminée")

    existing_attendance = getAttendanceForBlock(supabase, block_id)
    cap = cv2.VideoCapture(1)
    print("Webcam démarrée")

    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la webcam")
        return

    last_block_check_time = datetime.now()
    block_change_check_interval = timedelta(minutes=5)

    while True:
        if datetime.now() - last_block_check_time > block_change_check_interval:
            block_id = getActiveClassStudentsFaceData(supabase, env_vars['LOCAL'])[0]
            existing_attendance = getAttendanceForBlock(supabase, block_id)
            last_block_check_time = datetime.now()

        success, img = cap.read()
        if not success:
            print("Erreur de lecture de la webcam.")
            break

        recognize_faces(img, face_db, existing_attendance, supabase, block_id)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == "__main__":
    main()
