from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from utils.data_preparation import load_data
from utils.feature_extraction import get_face_embedding
import joblib
import numpy as np

# Load dataset
images, labels, class_names = load_data("data/dataset/")

# Generate face embeddings for training
embeddings = []
for image in images:
    embedding = get_face_embedding(image)
    embeddings.append(embedding)
embeddings = np.vstack(embeddings)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Train SVM classifier
svm = SVC(kernel='linear', probability=True)
svm.fit(embeddings, labels)

# Save the trained model and label encoder
joblib.dump(svm, "models/svm_classifier.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")