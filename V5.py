import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
import numpy as np

# Modèle de reconnaissance le plus performant
models = ["Facenet512", "ArcFace", "Dlib", "VGG-Face"]
distance_metrics = ["cosine", "euclidean", "euclidean_l2"]

# Backend pour la détection des visages
detector_backend = 'retinaface'  # Peut être 'mtcnn', 'ssd', 'retinaface', 'opencv', etc.

# Chemin de l'image à analyser
img_path = "images/groupe-besoin.jpg"  # Remplace par le chemin de ton image

# Fonction pour convertir l'image en format `uint8`
def convert_image_to_uint8(image):
    """
    Convertir une image en type uint8 si nécessaire.
    """
    if image.dtype == 'float64':  # Si l'image est de type float64, la convertir en uint8
        image = np.uint8(image * 255)
    return image

# Étape 1 : Détection et affichage des visages détectés
print("Étape 1 : Détection des visages...")
faces_detected = DeepFace.extract_faces(img_path=img_path, detector_backend=detector_backend, align=True)

# Afficher tous les visages détectés
for i, face_obj in enumerate(faces_detected):
    face_image = face_obj["face"]
    print(f"Visage détecté {i + 1} - Dimensions : {face_image.shape}")
    
    # Convertir l'image si nécessaire
    face_image = convert_image_to_uint8(face_image)
    
    # Affichage du visage détecté
    plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Visage {i + 1}")
    plt.axis('off')
    plt.show()

# Étape 2 : Analyse des attributs faciaux (âge, genre, émotion, race)
print("Étape 2 : Analyse des attributs faciaux...")
attributes = DeepFace.analyze(img_path=img_path, actions=['age', 'gender', 'race', 'emotion'], detector_backend=detector_backend)

# Si plusieurs visages sont détectés, 'attributes' sera une liste. Sinon, ce sera un dictionnaire unique.
if isinstance(attributes, list):
    print(f"{len(attributes)} visages détectés pour l'analyse des attributs.")
    for idx, attr in enumerate(attributes):
        print(f"\n--- Visage {idx + 1} ---")
        print(f"Prédiction d'âge : {attr['age']}")
        print(f"Prédiction de genre : {attr['gender']}")
        print(f"Prédiction de race : {attr['dominant_race']}")
        print(f"Prédiction d'émotion : {attr['dominant_emotion']}")

        # Affichage des attributs faciaux prédits pour chaque visage
        image_to_show = convert_image_to_uint8(cv2.imread(img_path))  # Convertir si nécessaire
        plt.imshow(cv2.cvtColor(image_to_show, cv2.COLOR_BGR2RGB))
        plt.title(f"Visage {idx + 1} - Age : {attr['age']}, Genre : {attr['gender']}, Race : {attr['dominant_race']}, Émotion : {attr['dominant_emotion']}")
        plt.axis('off')
        plt.show()
else:
    # Cas où un seul visage est détecté
    print(f"Prédiction d'âge : {attributes['age']}")
    print(f"Prédiction de genre : {attributes['gender']}")
    print(f"Prédiction de race : {attributes['dominant_race']}")
    print(f"Prédiction d'émotion : {attributes['dominant_emotion']}")

    # Affichage des attributs faciaux prédits pour le visage détecté
    image_to_show = convert_image_to_uint8(cv2.imread(img_path))  # Convertir si nécessaire
    plt.imshow(cv2.cvtColor(image_to_show, cv2.COLOR_BGR2RGB))
    plt.title(f"Age : {attributes['age']}, Genre : {attributes['gender']}, Race : {attributes['dominant_race']}, Émotion : {attributes['dominant_emotion']}")
    plt.axis('off')
    plt.show()


# Étape 3 : Génération d'embeddings pour l'image
print("Étape 3 : Représentation faciale (Embeddings)...")
for model in models:
    print(f"\nModèle : {model}")
    embeddings = DeepFace.represent(img_path=img_path, model_name=model, detector_backend=detector_backend)
    
    # Affichage des dimensions des embeddings pour chaque modèle
    for embedding_obj in embeddings:
        embedding = embedding_obj["embedding"]
        print(f"Dimensions de l'embedding pour {model} : {len(embedding)}")

# Étape 4 : Vérification de correspondance (comparaison avec une autre image)
print("Étape 4 : Vérification de correspondance...")
# Image de comparaison (doit être changée pour une image appropriée)
img_comparison_path = "images/vlad.jpg"  # Remplace par le chemin de l'image à comparer

# Vérification avec les modèles les plus performants et les métriques de distance
for model in models:
    for metric in distance_metrics:
        print(f"Vérification avec le modèle '{model}' et la métrique '{metric}'...")
        result = DeepFace.verify(img1_path=img_path, img2_path=img_comparison_path, model_name=model, distance_metric=metric, detector_backend=detector_backend)
        print(f"Résultat : {result['verified']} (Distance : {result['distance']:.4f}, Seuil : {result['threshold']:.4f})")
        
        # Affichage côte à côte des images comparées avec le résultat
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        axes[0].set_title("Image d'origine")
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(cv2.imread(img_comparison_path), cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Image de comparaison\nRésultat : {'Même personne' if result['verified'] else 'Personne différente'}")
        axes[1].axis('off')
        
        plt.suptitle(f"Modèle : {model}, Métrique : {metric}")
        plt.show()