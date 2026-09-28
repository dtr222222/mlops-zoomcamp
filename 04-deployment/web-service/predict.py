import pickle
from flask import Flask, request, jsonify

with open('lin_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

def prepare_features(data):
    features = {}
    features['PU_DO'] = str(data['PULocationID']) + '_' + str(data['DOLocationID'])
    features['trip_distance'] = data['trip_distance']
    return features

def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds[0]

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
