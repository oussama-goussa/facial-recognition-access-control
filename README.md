# 🔐 AI-Powered Face Recognition Access Control System

*Real-time face recognition and access control using SVM, OpenCV, and Arduino communication*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange)
![Arduino](https://img.shields.io/badge/Arduino-Compatible-brightgreen)
![Virtual Serial](https://img.shields.io/badge/Virtual_Serial-Port-blueviolet)

## 📖 Overview

This project implements an intelligent access control system that performs real-time face recognition using Support Vector Machines (SVM) and Histogram of Oriented Gradients (HOG) features. The system provides secure authentication with visual feedback and serial communication for hardware integration.

## 🎯 Key Features

- **🤖 AI-Powered Recognition**: SVM model with HOG features for accurate face classification
- **⏱️ Real-time Detection**: Live video processing with OpenCV
- **👥 User Management**: Intuitive GUI for adding new users and training models
- **🔌 Serial Communication**: Arduino integration for hardware control
- **📊 Confidence Thresholding**: 80%+ confidence requirement for recognition
- **🎨 Modern GUI**: Tkinter interface with ttkbootstrap styling

## 🔗 Virtual Serial Setup

### Required Software
- **Virtual Serial Port Driver**: [Download here](https://www.virtual-serial-port.org/)
- **Wokwi Simulation**: [Live Simulation](https://wokwi.com/projects/419729501675955201)

### Configuration Steps

1. **Install Virtual Serial Port Driver**
   - Download and install from the official website
   - Create a virtual port pair (e.g., COM14 ↔ COM15)

2. **Python Serial Configuration**
```python
# In detection.py
SERIAL_PORT = "COM15"
try:
    ser = serial.Serial(SERIAL_PORT, 9600, timeout=1)
    time.sleep(2)  # Wait for initialization
    print("✅ Serial connection established.")
except Exception as e:
    print(f"⚠️ Serial connection error: {e}")
    ser = None
```

3. **Arduino Communication Protocol**
```cpp
// Example Arduino code to receive commands
void setup() {
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  if (Serial.available()) { // Vérifie si un nom est reçu via le port série
    receivedName = Serial.readStringUntil('\n'); // Lire la ligne complète
    receivedName.trim(); // Supprimer les espaces et sauts de ligne
    nameReceived = true;

    if (receivedName != "Inconnu" && receivedName != "Face the camera") {
        ouvrirPorte();
        afficherMessageBienvenue(receivedName);
    } else if (receivedName == "Inconnu") {  // Correction du guillemet
        fermerPorte();
        afficherMessageAccesRefuse();
    } else if (receivedName == "Face the camera") {  // Correction du else
        fermerPorte();
        afficherMessageAttente();
    }
  }

  // Si aucun nom n'a été reçu pendant longtemps, demander à faire face à la caméra
  if (!nameReceived) {
    afficherMessageAttente();
  }
}
```

## 🛠️ System Architecture

```
Live Camera → Face Detection → HOG Feature Extraction → SVM Classification → Serial Output
      ↑              ↓                   ↓                     ↓              ↓
   OpenCV       Haar Cascade        Feature Vector        Recognition     Arduino Control
```

## 📁 Project Structure

```
face-recognition-access-control/
├── models/
│   └── svm_model.pkl
├── dataset/
│   ├── user1/
│   │   ├── 0.jpg
│   │   └── ... (30 images per user)
│   └── user2/
├── src/
│   ├── detection.py              # Main recognition script
│   └── entrainement.py           # Training and user management GUI
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- **Hardware**: Webcam, Arduino board (optional)
- **Software**: 
  - Python 3.8+
  - OpenCV, scikit-learn, ttkbootstrap
  - Virtual Serial Port Driver
  - Wokwi account for simulation

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/oussama-goussa/facial-recognition-access-control.git
cd face-recognition-access-control
```

2. **Install Python dependencies**
```bash
pip install opencv-python scikit-learn ttkbootstrap pillow pyserial numpy
```

3. **Setup Project Structure**
```bash
mkdir -p models dataset
```

4. **Configure Virtual Serial Port**
   - Install Virtual Serial Port Driver
   - Create a pair: COM14 (Wokwi) ↔ COM15 (Python)
   - Update `SERIAL_PORT` in `detection.py` if needed

5. **Run the System**

   **Step 1: Add Users and Train Model**
   ```bash
   python entrainement.py
   ```
   - Click "➕ Ajouter Nouvel Utilisateur" to add new users
   - Capture 30 face images per user
   - Click "🎓 Entraîner le Modèle SVM" to train the model

   **Step 2: Start Recognition**
   ```bash
   python detection.py
   ```

## 🔧 Technical Details

### Face Recognition Pipeline

1. **Face Detection**: Haar Cascade classifier
2. **Feature Extraction**: HOG (Histogram of Oriented Gradients)
3. **Classification**: SVM with linear kernel
4. **Confidence Threshold**: 80% minimum for recognition

### HOG Feature Extraction
```python
def extract_hog_features(img):
    hog = cv2.HOGDescriptor()
    resized_img = cv2.resize(img, (128, 128))
    return hog.compute(resized_img).flatten()
```

### SVM Training
```python
self.svm_model = SVC(kernel='linear', probability=True)
self.svm_model.fit(X, y_encoded)
```

## 🎮 User Interface

### Main Application (`entrainement.py`)
- **Modern Tkinter GUI** with ttkbootstrap styling
- **User Management**: Add new users with face capture
- **Model Training**: Train SVM classifier with collected data
- **Real-time Preview**: Live video feed during capture

### Recognition Module (`detection.py`)
- **Real-time Video Processing**: OpenCV video capture
- **Visual Feedback**: Bounding boxes and recognition labels
- **Serial Communication**: Send recognition results to Arduino
- **Confidence-based Decisions**: Only accept high-confidence matches

## 🔌 Serial Communication Protocol

### Data Format
The system sends recognized names via serial every 5 seconds:
- `"Authorized User"` - When recognized with >80% confidence
- `"Inconnu"` - When unknown or low confidence
- `"Face the camera"` - When no face is detected

### Rate Limiting
```python
def envoyer_serial(message):
    global last_sent_time
    current_time = time.time()
    
    if current_time - last_sent_time >= 5:  # Send every 5 seconds
        last_sent_time = current_time
        if ser:
            ser.write(f"{message}\n".encode())
```

## 🛠️ Troubleshooting

### Common Issues

1. **Camera Not Working**
   - Check if webcam is connected and not used by other applications
   - Verify OpenCV installation: `python -c "import cv2; print(cv2.__version__)"`

2. **Model Training Errors**
   - Ensure at least one user has been added with 30 images
   - Check dataset directory structure
   - Verify scikit-learn version compatibility

3. **Serial Connection Failed**
   - Confirm virtual port pairing
   - Check port permissions (Windows: Device Manager)
   - Verify baud rate matching (9600)

4. **Recognition Accuracy Issues**
   - Ensure good lighting during face capture
   - Capture faces from different angles
   - Retrain model with more diverse images

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Recognition Accuracy | > 80% |
| Processing Speed | Real-time (30 FPS) |
| Face Detection | Haar Cascade |
| Feature Extraction | HOG (128×128) |
| Classification | SVM with Linear Kernel |
| Confidence Threshold | 80% |

## 🔒 Security Features

- **Confidence Thresholding**: Prevents false positives
- **Continuous Monitoring**: Real-time face detection
- **Access Logging**: Serial communication for audit trails
- **User Management**: Secure addition of authorized users

## 🔗 Useful Links

- **GitHub Repository**: [https://github.com/your-username/face-recognition-access-control](https://github.com/your-username/face-recognition-access-control)
- **Wokwi Simulation**: [https://wokwi.com/projects/419729501675955201](https://wokwi.com/projects/419729501675955201)
- **Virtual Serial Port**: [https://www.virtual-serial-port.org/](https://www.virtual-serial-port.org/)
- **OpenCV Documentation**: [https://docs.opencv.org/](https://docs.opencv.org/)

## 🎓 Educational Value

This project demonstrates:
- Machine learning for computer vision
- Real-time video processing
- Serial communication protocols
- GUI development with Tkinter
- SVM classification techniques
- HOG feature extraction

## ⚠️ Disclaimer

This system is designed for educational and demonstration purposes. For production use, consider additional security measures such as:
- Liveness detection to prevent spoofing
- Encryption of stored face data
- Multi-factor authentication
- Regular model updates and monitoring

---

<div align="center">

**Made with ❤️ for Secure Access Control**

*If this project helps you, please give it a ⭐!*

[![GitHub stars](https://img.shields.io/github/stars/oussama-goussa/facial-recognition-access-control?style=social)](https://github.com/oussama-goussa/facial-recognition-access-control)

**🔒 Secure • 🤖 Intelligent • 🚀 Real-time**

</div>

---

*For questions and support, please open an issue on GitHub.*
