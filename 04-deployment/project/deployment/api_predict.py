from flask import Flask, jsonify, request
import mlflow

# 模型直接导入，随脚本运行而打开，随脚本结束而销毁。
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('taxi_duration_prediction')

model_uri = "models:/m-89cf2bc8adea4f68ae9e961cdfd5d809"
pipeline = mlflow.sklearn.load_model(model_uri)

app = Flask('duration_predictor')

def prepare_features(data):
    features = {}
    features['PU_DO'] = str(data['PULocationID']) + '_' + str(data['DOLocationID'])
    features['trip_distance'] = data['trip_distance']
    return features

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    features = prepare_features(data)

    preds = pipeline.predict([features])
    result = { "duration": preds[0] }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
