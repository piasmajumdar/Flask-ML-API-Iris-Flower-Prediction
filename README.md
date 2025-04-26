# Flask-ML-API-Iris-Flower-Prediction
This project trains a simple Decision Tree Classifier on the Iris dataset and serves predictions via a Flask API. Send flower measurements to the /predict endpoint to get a species classification (Setosa, Versicolor, or Virginica).

🚀 Features:
ML model trained with scikit-learn

API endpoint: POST /predict
Input: JSON with 4 features
Output: Predicted flower class

🛠 Tech Stack:
Python
Flask
scikit-learn
joblib

✅ How to Run:
pip install flask scikit-learn joblib
python train_model.py   # Trains and saves the model
python app.py           # Starts the Flask API
