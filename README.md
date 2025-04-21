# LieDetectionUsingML
# 🤥 Lie Detection using Facial Muscle Movements

This project aims to detect whether a person is lying or telling the truth by analyzing **facial muscle expressions** using **MediaPipe FaceMesh**, **OpenCV**, and **machine learning**. The system is trained on facial landmark data extracted from labeled images and can make predictions using real-time webcam input.

---

## 📁 Dataset Structure

The dataset is divided into two categories:

LieDetectionDataset/ ├── Train/ │ ├── Lie/ │ │ ├── Person1/ │ │ │ ├── Question1/ │ │ │ └── Question2/ │ └── Truth/ │ ├── Person2/ │ │ ├── Question1/ │ │ └── Question2/ └── Test/ ├── Lie/ └── Truth/


Each question folder contains image frames (e.g., `.jpg`, `.png`) extracted from video interviews.

---

## 🧠 Workflow

### 1. **Facial Landmark Extraction**

- **MediaPipe FaceMesh** is used to extract 468 facial landmarks (x, y, z) per image.
- These are flattened into a single vector of 1404 values per image.
- An additional column for `label` (`1` = Lie, `0` = Truth) and `image_path` is added.
- The final CSV has shape: `(n_samples, 1406)`.

```python
[x_0, y_0, z_0, x_1, y_1, z_1, ..., x_467, y_467, z_467, label, image_path]

train_landmarks.csv is created from the training dataset.

test_landmark_data.csv is created from the test dataset.

2. Data Preprocessing
image_path column is dropped before modeling.

Features (X) and labels (y) are separated.

The dataset is split into 80% train and 20% validation using train_test_split.

Features are standardized using StandardScaler.

3. Model Training
A Logistic Regression model is trained on the scaled landmarks.

Performance is evaluated using:

Confusion Matrix

Accuracy

Classification Report

Example Result:

mathematica
Copy
Edit
Accuracy: 76.75%
Precision (Truth): 0.77
Recall (Lie): 0.75
4. Live Prediction using Webcam
A custom image is captured via webcam using OpenCV.

MediaPipe extracts facial landmarks from the image.

The same StandardScaler is used to transform the input.

The trained model predicts whether the expression indicates a Lie or Truth.

🧪 Requirements
bash
Copy
Edit
pip install opencv-python mediapipe scikit-learn pandas numpy matplotlib tqdm seaborn
🚀 How to Run
1. Extract Landmarks
python
Copy
Edit
# Extract landmarks from training data
# Creates train_landmarks.csv
2. Train Model
python
Copy
Edit
# Preprocess data, scale features
# Train logistic regression model
3. Predict Using Webcam
python
Copy
Edit
# Open webcam
# Capture image on pressing 'c'
# Extract features and predict lie/truth
🧠 Future Enhancements
Use deep learning (CNNs or LSTMs) for temporal analysis from videos.

Include body language and voice tone as features.

Enhance dataset diversity and quality.

🙌 Acknowledgements
MediaPipe by Google

Scikit-learn

OpenCV

👤 Author
Nitish Paramanik
Machine Learning Research Enthusiast
“Because the face never lies, unless it does.”

📸 Sample Demo (optional)
You can insert a demo GIF or image of prediction output here.

📂 Output Files
train_landmarks.csv

test_landmark_data.csv

captured_image.jpg (temporary)
