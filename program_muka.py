import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Ukuran standar wajah untuk PCA
face_size = (128, 128)

def load_image(image_path):
    """Load image, return BGR image and grayscale."""
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def detect_faces(gray, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """Detect faces using Haar Cascade."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    return cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )

def crop_faces(gray, faces, return_all=False):
    """Crop detected faces; default: only largest face."""
    crops, coords = [], []
    if len(faces) > 0:
        if return_all:
            for (x, y, w, h) in faces:
                crops.append(gray[y:y+h, x:x+w])
                coords.append((x, y, w, h))
        else:
            # pilih wajah dengan area terbesar
            x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
            crops.append(gray[y:y+h, x:x+w])
            coords.append((x, y, w, h))
    return crops, coords

def resize_and_flatten(face):
    """Resize ke face_size dan flatten menjadi 1D array."""
    resized = cv2.resize(face, face_size)
    return resized.flatten()

class MeanCentering(BaseEstimator, TransformerMixin):
    """Subtract mean face from each sample."""
    def fit(self, X, y=None):
        self.mean_face_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.mean_face_

def build_pipeline():
    """Buat pipeline sklearn: centering → PCA → SVM"""
    return Pipeline([
        ('center', MeanCentering()),
        ('pca', PCA(svd_solver='randomized', whiten=True, random_state=177)),
        ('svc', SVC(kernel='linear', probability=True, random_state=177))
    ])

def train_model(dataset_dir, model_path):
    """Load dataset, train pipeline, evaluasi, dan simpan."""
    X, y = [], []
    for root, _, files in os.walk(dataset_dir):
        label = os.path.basename(root)
        for fname in files:
            _, gray = load_image(os.path.join(root, fname))
            if gray is None:
                continue
            faces = detect_faces(gray)
            crops, _ = crop_faces(gray, faces)
            if not crops:
                continue
            X.append(resize_and_flatten(crops[0]))
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=177
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluasi
    y_pred = pipeline.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Simpan model
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {model_path}")
    return pipeline

def eigenface_prediction(pipeline, gray):
    """Lakukan prediksi untuk frame grayscale."""
    faces = detect_faces(gray)
    crops, coords = crop_faces(gray, faces, return_all=True)
    if not crops:
        return [], [], []
    Xf = np.array([resize_and_flatten(c) for c in crops])
    labels = pipeline.predict(Xf)
    scores = pipeline.decision_function(Xf)  
    max_scores = np.max(scores, axis=1)
    return max_scores, labels, coords

def draw_result(frame, scores, labels, coords):
    """Gambarkan kotak dan label di frame warna."""
    out = frame.copy()
    for (x, y, w, h), lbl, sc in zip(coords, labels, scores):
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{lbl} ({sc:.2f})"
        cv2.putText(out, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    return out

def real_time_recognition(pipeline):
    """Mulai webcam, tampilkan hasil deteksi & pengenalan."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scores, labels, coords = eigenface_prediction(pipeline, gray)
        result = draw_result(frame, scores, labels, coords)
        cv2.imshow('Real-Time Recognition', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    DATASET_DIR = './images'
    MODEL_PATH = 'eigenface_pipeline.pkl'

    if not os.path.exists(MODEL_PATH):
        pipeline = train_model(DATASET_DIR, MODEL_PATH)
    else:
        print(f"Loading model from {MODEL_PATH}...")
        with open(MODEL_PATH, 'rb') as f:
            pipeline = pickle.load(f)

    real_time_recognition(pipeline)