from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('iris_tree.joblib')
target_names = ['setosa', 'versicolor', 'virginica']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]
    pred_idx = model.predict([features])[0]
    return jsonify({'prediction': target_names[pred_idx]})

if __name__ == '__main__':
    app.run(debug=True)
