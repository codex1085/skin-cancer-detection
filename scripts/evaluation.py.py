from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load model
model = load_model('models/skin_cancer_model.h5')

# Load test data
test_images = np.load('data/test_images.npy')
test_labels = np.load('data/test_labels.npy')

# Evaluate
predictions = model.predict(test_images)
predicted_classes = (predictions > 0.5).astype("int32")

# Metrics
print("Confusion Matrix:")
print(confusion_matrix(test_labels, predicted_classes))

print("\nClassification Report:")
print(classification_report(test_labels, predicted_classes))
