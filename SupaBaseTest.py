from supabase import create_client, Client
from datetime import datetime, timezone
import json

url: str = "https://onhcyopzqgihsonkdwcb.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9uaGN5b3B6cWdpaHNvbmtkd2NiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mjc2MTc5MjAsImV4cCI6MjA0MzE5MzkyMH0.r_aIPYVr2P5i04Lkj7DNYSzSc7ccoVkLDZcW-Id1Y3I"
supabase: Client = create_client(url, key)

student_email = 'gaetan.carbonnelle1@gmail.com'
block_id = 4
status = 'Present'
timestamp = datetime.now(timezone.utc).isoformat()


reponse = supabase.rpc("get_active_class_students_face_data", {"local_now":"L217"}).execute()

print(reponse.data["block_id"])
print(reponse.data["students"])


try:
    # Envois la présence de l'étudiant dans DB 
    response = supabase.rpc('insert_new_attendance', {
        "attendance_student_email": student_email,
        "attendance_block_id": block_id,
        "attendance_status": status,
        "attendance_timestamp": timestamp
    }).execute()

    # véfirie le retours de la DB
    if response.data:
        print(f"l'étudants {student_email} à bien été enregistré sur le block_ID N°{block_id}")
    else:
        print(f"L'étudiant {student_email} à déja été enregistré sur le block_ID N°{block_id}")

except Exception as e:
    print(f"An error occurred: {e}")
