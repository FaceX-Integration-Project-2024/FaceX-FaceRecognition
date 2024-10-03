from deepface import DeepFace
import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
face_cascade_path = './env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
eye_cascade_path = './env/Lib/site-packages/cv2/data/haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# Chemins des images à comparer (Remplacer par tes chemins d'images)
img1_path = "images/fredW.jpg"  # Remplace par le chemin de l'image 1
img2_path = "images/George2.png"  # Remplace par le chemin de l'image 2

# Définir le modèle et la métrique
model_name = "Facenet512"
distance_metric = "cosine"

# Effectuer la comparaison
print(f"Comparaison en utilisant le modèle '{model_name}' et la métrique '{distance_metric}'...\n")
result = DeepFace.verify(
    img1_path=img1_path,
    img2_path=img2_path,
    model_name=model_name,
    distance_metric=distance_metric,
    enforce_detection=False  # Détection déjà faite, on peut ignorer cela pour gagner du temps
)

# Ajuster le seuil manuellement (valeur plus stricte)
custom_threshold = 0.3  # Seuil de distance personnalisé pour plus de fiabilité

# Afficher le résultat avec le seuil personnalisé
print(f"Résultat de la comparaison : {'Même personne' if result['distance'] < custom_threshold else 'Personnes différentes'}")
print(f"Distance calculée : {result['distance']:.4f}")
print(f"Seuil personnalisé : {custom_threshold:.4f}")
