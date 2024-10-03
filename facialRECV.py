import cv2
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace

# Définir les chemins des images
img1_path = "images/bide.jpg"

try:
    # Extraire les visages de l'image
    face_objs = DeepFace.extract_faces(
        img_path=img1_path,
        detector_backend='mtcnn',
        align=True
    )
    print(f"Nombre de visages détectés : {len(face_objs)}")

    # Afficher les informations sur le premier visage extrait
    if len(face_objs) > 0:
        # Récupérer l'image du visage et ses informations
        face_image = face_objs[0]["face"]
        print(f"Dimensions de l'image du visage : {face_image.shape}")
        print(f"Type de l'image : {face_image.dtype}")

        # Convertir l'image de `float64` à `uint8`
        if face_image.dtype == 'float64':
            # Normaliser l'image (de 0 à 255) et convertir en `uint8`
            face_image = np.uint8(face_image * 255)
        
        # Afficher le visage extrait
        plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        plt.title("Visage détecté et extrait")
        plt.axis('off')  # Cacher les axes
        plt.show()

    # Afficher les informations faciales dans un DataFrame
    face_data = [face_obj["facial_area"] for face_obj in face_objs]
    df = pd.DataFrame(face_data)
    print("Informations sur les visages détectés :")
    print(df)

except Exception as e:
    print(f"Erreur lors de l'extraction des visages : {e}")
