import cv2

# Test d'ouverture des fichiers Haar Cascade
face_cascade_path = './env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
eye_cascade_path = './env/Lib/site-packages/cv2/data/haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

if face_cascade.empty():
    print("Erreur : Impossible d'ouvrir le fichier haarcascade_frontalface_default.xml")
else:
    print("Fichier haarcascade_frontalface_default.xml ouvert avec succès")

if eye_cascade.empty():
    print("Erreur : Impossible d'ouvrir le fichier haarcascade_eye.xml")
else:
    print("Fichier haarcascade_eye.xml ouvert avec succès")