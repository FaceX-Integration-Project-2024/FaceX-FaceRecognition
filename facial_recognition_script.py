from deepface import DeepFace
import cv2

import pandas as pd
import matplotlib.pyplot as plt

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]

alignment_modes = [True, False]
# Vérifier le chargement des images
img1_path = "images/bide.jpg"
img2_path = "images/unknow.jpg"

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None:
    raise ValueError(f"L'image '{img1_path}' n'a pas pu être chargée. Vérifiez le chemin de l'image.")
if img2 is None:
    raise ValueError(f"L'image '{img2_path}' n'a pas pu être chargée. Vérifiez le chemin de l'image.")
else:
    print(f"Les images '{img1_path}' et '{img2_path}' ont été chargées avec succès.")

# Vérifier les visages avec DeepFace
try:
    # Utilisation de 'ssd' comme backend de détection
    result = DeepFace.verify(img1_path, img2_path, enforce_detection=False)
    print(f"Est-ce la même personne ? {result['verified']}")
    print(result)
    face_objs = DeepFace.extract_faces(
        img_path = img1_path, 
        detector_backend = backends[4],
        align = alignment_modes[0],
    )
    print(face_objs)
   
except Exception as e:
    print(f"Erreur lors de la vérification des visages : {e}")
