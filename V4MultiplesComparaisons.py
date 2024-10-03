import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deepface import DeepFace

# Définir les chemins des images
img1_path = "images/groupe-besoin.jpg"
img2_path = "images/vlad.jpg"

def load_and_check_image(path):
    """
    Fonction utilitaire pour charger et vérifier l'intégrité de l'image.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"L'image '{path}' n'a pas pu être chargée. Vérifiez le chemin de l'image.")
    return img

def extract_and_display_face(img_path, backend='mtcnn'):
    """
    Extraire et afficher le visage détecté dans l'image.
    Retourne le visage extrait en format `uint8`.
    """
    # Extraire les visages
    face_objs = DeepFace.extract_faces(
        img_path=img_path,
        detector_backend=backend,
        align=True
    )
    if len(face_objs) == 0:
        raise ValueError(f"Aucun visage détecté dans l'image {img_path}.")
    
    # Récupérer l'image du visage
    face_image = face_objs[0]["face"]  # Extraire seulement le premier visage
    print(f"Visage détecté dans '{img_path}', Dimensions : {face_image.shape}, Type : {face_image.dtype}")

    # Convertir en `uint8` si nécessaire
    if face_image.dtype == 'float64':
        face_image = np.uint8(face_image * 255)

    # Afficher le visage extrait
    plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Visage extrait de {img_path}")
    plt.axis('off')  # Masquer les axes
    plt.show()

    return face_image

# Charger les images et vérifier leur intégrité
img1 = load_and_check_image(img1_path)
img2 = load_and_check_image(img2_path)

# Afficher les deux visages extraits
print("Extraction et affichage des visages...")
face1 = extract_and_display_face(img1_path, backend='mtcnn')
face2 = extract_and_display_face(img2_path, backend='mtcnn')

# Comparer les deux images pour vérifier s'il s'agit de la même personne
try:
    result = DeepFace.verify(img1_path, img2_path,model_name='Facenet512',detector_backend='mtcnn', enforce_detection=False,distance_metric='cosine')
    print(f"Est-ce la même personne ? {result['verified']}")
    print("Détails de la comparaison :")
    print(result)

    # Afficher les deux visages côte à côte avec le résultat de la comparaison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Deux images côte à côte

    # Afficher le premier visage
    axes[0].imshow(cv2.cvtColor(face1, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Visage 1 extrait (image 1)")
    axes[0].axis('off')

    # Afficher le second visage
    axes[1].imshow(cv2.cvtColor(face2, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Visage 2 extrait (image 2)")
    axes[1].axis('off')

    # Ajouter un titre global
    fig.suptitle(f"Résultat de la comparaison : {'Même personne' if result['verified'] else 'Personnes différentes'}", fontsize=16)
    plt.show()

except Exception as e:
    print(f"Erreur lors de la vérification des visages : {e}")
