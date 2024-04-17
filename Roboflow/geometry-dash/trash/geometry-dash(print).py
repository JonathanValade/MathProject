import pyautogui
import cv2
import numpy as np
from inference import get_model
import supervision as sv

# Obtenez la position et la taille de la fenêtre de jeu
game_window = pyautogui.getWindowsWithTitle('Geometry Dash')[0]
left, top, width, height = game_window.left, game_window.top, game_window.width, game_window.height

# Chargez un modèle YOLOv8 pré-entraîné
ROBOFLOW_API_KEY_Jo_Mayo = "I3OqXk01rMiMxjUj1zoj"
model = get_model(model_id="projet-math-geometry-dash/2", api_key=ROBOFLOW_API_KEY_Jo_Mayo)

# Créez une instance de BoundingBoxAnnotator pour les annotations en temps réel
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Initialisation du compteur
result_number = 0

# Boucle pour la capture vidéo en temps réel et l'annotation
while True:
    # Capturez la partie de l'écran correspondant à la fenêtre du jeu
    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    frame = np.array(screenshot)

    # Convertissez le format de l'image de RGB à BGR (car OpenCV lit les images en BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Exécutez l'inférence sur l'image
    results = model.infer(frame)

    # Incrémenter le compteur
    result_number += 1

    # Imprimer les résultats de l'inférence dans le terminal avec le numéro
    print(f"Results for frame {result_number}:")
    for prediction in results[0].predictions:
        print(f"Object: {prediction.class_name}, Confidence: {prediction.confidence}, X,Y: {prediction.x, prediction.y}")

# Libérer les ressources
cv2.destroyAllWindows()