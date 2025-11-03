import os
import cv2
import pickle
import numpy as np
import ttkbootstrap as ttkb
from tkinter import messagebox
from PIL import Image, ImageTk
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from threading import Thread
from tkinter import messagebox, LEFT, RIGHT


class FaceRecognitionSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Système de Gestion Utilisateurs")
        self.root.geometry("600x200")
        self.style = ttkb.Style()
        self.style.theme_use('solar')

        # Configuration SVM
        self.svm_model = None
        self.le = LabelEncoder()

        # Configuration OpenCV
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Interface utilisateur
        self.create_widgets()
        self.capturing = False

        # Vérification du dossier dataset
        self.DATASET_PATH = "dataset"
        if not os.path.exists(self.DATASET_PATH):
            os.makedirs(self.DATASET_PATH)

    def create_widgets(self):
        main_frame = ttkb.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        title_label = ttkb.Label(main_frame, text="Gestion des Utilisateurs", font=('Helvetica', 16, 'bold'))
        title_label.pack(pady=15)

        control_frame = ttkb.Frame(main_frame)
        control_frame.pack(pady=20)

        self.btn_add_user = ttkb.Button(
            control_frame,
            text="➕ Ajouter Nouvel Utilisateur", 
            command=self.show_add_user_dialog,
            bootstyle="primary",
            width=25,
            padding=(10, 5)
        )
        self.btn_add_user.pack(side=LEFT, padx=15)

        self.btn_train = ttkb.Button(
            control_frame,
            text="🎓 Entraîner le Modèle SVM", 
            command=self.train_model,
            bootstyle="info",
            width=25,
            padding=(10, 5)
        )
        self.btn_train.pack(side=LEFT, padx=15)

    def show_add_user_dialog(self):
        self.add_user_window = ttkb.Toplevel(self.root)
        self.add_user_window.title("Nouvel Utilisateur")
        self.add_user_window.geometry("600x500")

        ttkb.Label(self.add_user_window, text="Nom de l'utilisateur:", font=('Helvetica', 10)).pack(pady=10)

        self.user_entry = ttkb.Entry(self.add_user_window, width=25)
        self.user_entry.pack(pady=5)

        self.video_label = ttkb.Label(self.add_user_window)
        self.video_label.pack(pady=10)

        self.progress_label = ttkb.Label(self.add_user_window, text="Prêt à capturer...")
        self.progress_label.pack(pady=5)

        btn_frame = ttkb.Frame(self.add_user_window)
        btn_frame.pack(pady=15)

        ttkb.Button(btn_frame, text="▶ Commencer la Capture", command=self.start_capture, bootstyle="success", width=22).pack(side=LEFT, padx=10)
        ttkb.Button(btn_frame, text="✖ Annuler", command=self.cancel_capture, bootstyle="danger", width=22).pack(side=RIGHT, padx=10)

    def start_capture(self):
        user = self.user_entry.get().strip()
        if not user:
            messagebox.showerror("Erreur", "Veuillez entrer un utilisateur")
            return

        user_dir = os.path.join(self.DATASET_PATH, user)
        if os.path.exists(user_dir):
            messagebox.showerror("Erreur", "Utilisateur déjà existant")
            return

        os.makedirs(user_dir)
        self.capturing = True
        Thread(target=self.update_video_feed, daemon=True).start()
        Thread(target=lambda: self.add_user(user_dir), daemon=True).start()

    def update_video_feed(self):
        while self.capturing:
            ret, frame = self.cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                imgtk = ImageTk.PhotoImage(image=pil_image)

                self.video_label.configure(image=imgtk)
                self.video_label.image = imgtk

            cv2.waitKey(1)

    def add_user(self, user_dir):
        count = 0
        self.progress_label.config(text="Positionnez-vous devant la caméra...")

        while count < 30 and self.capturing:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    if count < 30:
                        cv2.imwrite(f"{user_dir}/{count}.jpg", gray[y:y+h, x:x+w])
                        count += 1
                        self.progress_label.config(text=f"Capture : {count}/30")
                        cv2.waitKey(300)

        self.capturing = False
        if count >= 30:
            messagebox.showinfo("Succès", "Utilisateur ajouté")
        else:
            messagebox.showwarning("Interrompu", "Capture annulée")
            os.rmdir(user_dir)

    def cancel_capture(self):
        self.capturing = False
        self.add_user_window.destroy()

    def extract_hog_features(self, img):
        hog = cv2.HOGDescriptor()
        resized_img = cv2.resize(img, (128, 128))
        return hog.compute(resized_img).flatten()

    def train_model(self):
        try:
            X, y = [], []
            for user_id in os.listdir(self.DATASET_PATH):
                user_dir = os.path.join(self.DATASET_PATH, user_id)
                if os.path.isdir(user_dir):
                    for image_name in os.listdir(user_dir):
                        img_path = os.path.join(user_dir, image_name)
                        img = cv2.imread(img_path, 0)
                        if img is None:
                            continue
                        features = self.extract_hog_features(img)
                        X.append(features)
                        y.append(user_id)

            if len(X) == 0:
                messagebox.showerror("Erreur", "Aucune donnée d'entraînement")
                return

            y_encoded = self.le.fit_transform(y)
            self.svm_model = SVC(kernel='linear', probability=True)
            self.svm_model.fit(X, y_encoded)

            with open("../models/svm_model.pkl", "wb") as f:
                pickle.dump((self.svm_model, self.le), f)

            messagebox.showinfo("Succès", "Modèle entraîné avec succès!")

        except Exception as e:
            messagebox.showerror("Erreur", f"Échec de l'entraînement : {str(e)}")

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = ttkb.Window()
    app = FaceRecognitionSystem(root)
    root.mainloop()