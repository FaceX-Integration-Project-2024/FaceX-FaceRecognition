import cv2
import pickle
import cvzone
import face_recognition
import numpy as np

# Ouvrir la webcam (indice 0 correspond généralement à la première webcam)
cap = cv2.VideoCapture(1)

print('Loading Encoded File...')
file = open('EncodeFile.p', 'rb')
encodeListKnownIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownIds
print('Encoded File Loaded')

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam")
else:
    while True:
        succes, img = cap.read()

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                print("Known Face Detected")
                print(studentIds[matchIndex])

        if not succes:
            print("Erreur : Impossible de lire l'image de la webcam")
            break

        cv2.imshow('Webcam', img)

        # Quitter en appuyant sur 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
