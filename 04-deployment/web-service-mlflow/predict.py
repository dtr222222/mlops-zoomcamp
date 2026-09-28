import pickle
import mlflow
from flask import Flask, request, jsonify

model_uri = 'runs:/622d5f916c7e4ed78a4b332fcb10a675/model'
model = mlflow.pyfunc.load_model(model_uri)

def prepare_features(data):
    features = {}
    features['PU_DO'] = str(data['PULocationID']) + '_' + str(data['DOLocationID'])
    features['trip_distance'] = data['trip_distance']
    return features

def predict(features):
    preds = model.predict(features)
    return float(preds[0])

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    features = prepare_features(data)
    pred = predict(features)
    result = {
        'duration': pred
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
