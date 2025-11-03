import cv2
import pickle
import numpy as np
import serial
import time

# Charger le modèle SVM et le LabelEncoder
with open("../models/svm_model.pkl", "rb") as f:
    svm_model, le = pickle.load(f)

# Vérifier et ouvrir le port série
SERIAL_PORT = "COM15"
try:
    ser = serial.Serial(SERIAL_PORT, 9600, timeout=1)
    time.sleep(2)  # Attendre l'initialisation
    print("✅ Connexion série établie.")
except Exception as e:
    print(f"⚠️ Erreur de connexion série: {e}")
    ser = None

# Initialiser la caméra et le classificateur de visages
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_hog_features(img):
    """Extrait les caractéristiques HOG d'une image redimensionnée."""
    hog = cv2.HOGDescriptor()
    resized_img = cv2.resize(img, (128, 128))
    return hog.compute(resized_img).flatten()

def envoyer_serial(message):
    """Envoie un message via le port série toutes les 3 secondes."""
    global last_sent_time
    current_time = time.time()
    
    if current_time - last_sent_time >= 5:  # Vérifie si 5 sec se sont écoulées
        last_sent_time = current_time
        if ser:
            try:
                ser.write(f"{message}\n".encode())
                ser.flush()  # S'assure que les données sont envoyées immédiatement
                print(f"📤 Envoyé : {message}")
            except Exception as e:
                print(f"⚠️ Erreur d'envoi série: {e}")

# Variable pour suivre le dernier envoi
last_sent_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:  # Si aucune face n'est détectée
        detected_name = "Face the camera"
        color = (0, 0, 255)  # Rouge pour le texte
    else:
        detected_name = "Inconnu"  # Valeur par défaut
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            features = extract_hog_features(face_img)
            proba = svm_model.predict_proba([features])[0]

            if np.max(proba) > 0.8:  # Seuil de confiance
                detected_name = le.inverse_transform([np.argmax(proba)])[0]
                color = (0, 255, 0)  # Vert si la personne est reconnue
            else:
                detected_name = "Inconnu"
                color = (0, 0, 255)  # Rouge si inconnu

            # Afficher le nom sur l’image
            cv2.putText(frame, detected_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # Envoyer le nom détecté toutes les 5 secondes
    envoyer_serial(detected_name)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()