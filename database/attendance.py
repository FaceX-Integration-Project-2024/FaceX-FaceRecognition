from datetime import datetime

def getActiveClassStudentsFaceData(supabase, local):
    response = supabase.rpc("get_active_class_students_face_data", {"local_now": local}).execute()
    return response.data["block_id"], response.data["students"]

def getAttendanceForBlock(supabase, class_block_id):
    response = supabase.rpc("get_attendance_for_class_block_python", {"class_block_id": int(class_block_id)}).execute()
    return {attendance['student_email'] for attendance in response.data}

def postStudentAttendanceDB(supabase, student_email, block_id, timestamp=None, status="Present"):
    if not timestamp:
        timestamp = datetime.now().isoformat()
    try:
        response = supabase.rpc('post_new_attendance', {
            "attendance_student_email": student_email,
            "attendance_block_id": block_id,
            "attendance_status": status,
            "attendance_timestamp": timestamp
        }).execute()

        if response.data:
            print(f"Présence enregistrée pour {student_email} (Block ID : {block_id}).")
        else:
            print(f"{student_email} était déjà enregistré pour ce bloc.")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement de la présence pour {student_email} : {e}")

