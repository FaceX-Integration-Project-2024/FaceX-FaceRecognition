import cv2
import json
import numpy as np
import face_recognition
import time
import multiprocessing

def visageReconosition(args):
    face_index, encodeFace, faceLoc = args
    top, right, bottom, left = faceLoc
    min_distance = float("inf")
    identified_person = None

    for person, embeddings in face_db.items():
        for embedding in embeddings:
            embedding = normalize(np.array(embedding))
            distance = np.linalg.norm(encodeFace - embedding)
            if distance < min_distance:
                min_distance = distance
                identified_person = person

    seuil_facerecognition = 0.6
    if min_distance < seuil_facerecognition:
        color = (0, 255, 0)
    else:
        identified_person = "Inconnu"
        color = (0, 0, 255)

    return identified_person, min_distance, faceLoc, color

def normalize(embedding):
    return embedding / np.linalg.norm(embedding)

with open('face_database_structured.json', 'r') as f:
    face_db = json.load(f)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    cap = cv2.VideoCapture(0)
    print("Webcam démarrée")

    if not cap.isOpened():
        print("Erreur : Impossible d'accéder à la webcam")
    else:
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
        process_every_n_frames = 5
        frame_count = 0
        previous_results = []

        while True:
            success, img = cap.read()
            if not success:
                print("Erreur de lecture de la webcam.")
                break

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if frame_count % process_every_n_frames == 0:
                small_frame = cv2.resize(img_rgb, (0, 0), fx=0.4, fy=0.4)
                facesCurFrame = face_recognition.face_locations(small_frame)
                encodesCurFrame = face_recognition.face_encodings(small_frame, facesCurFrame)
                task = [(face_index, encodeFace, faceLoc) for face_index, (encodeFace, faceLoc) in enumerate(zip(encodesCurFrame, facesCurFrame))]

                if len(task) > 1:
                    results = list(pool.imap_unordered(visageReconosition, task))
                else:
                    results = [visageReconosition(task[0])] if task else []

                previous_results = results
            else:
                results = previous_results

            for result in results:
                identified_person, distance, faceLoc, color = result
                top, right, bottom, left = faceLoc
                top, right, bottom, left = int(top * 2.5), int(right * 2.5), int(bottom * 2.5), int(left * 2.5)
                cv2.rectangle(img, (left, top), (right, bottom), color, 2)
                cv2.putText(img, f"{identified_person} - {distance:.2f}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow('Webcam', img)

            if cv2.waitKey(1) & 0xFF == ord('p'):
                print("Quitter...")
                break

            frame_count += 1

        pool.terminate()
        cap.release()
        cv2.destroyAllWindows()
