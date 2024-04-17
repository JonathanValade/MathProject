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

# Boucle pour la capture vidéo en temps réel et l'annotation
while True:
    # Capturez la partie de l'écran correspondant à la fenêtre du jeu
    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    frame = np.array(screenshot)

    # Convertissez le format de l'image de RGB à BGR (car OpenCV lit les images en BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Exécutez l'inférence sur l'image
    results = model.infer(frame)

    # Afficher les résultats de l'inférence dans le terminal
    print("Results for the current frame:")
    for label, confidence, box in results[0]:
        print(f"Label: {label}, Confidence: {confidence}, Bounding Box: {box}")

    # Sortir de la boucle si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cv2.destroyAllWindows()
