import base64
import json
import os
from flask import Flask, request, jsonify  # 🚨 引入 Flask

PREDICTION_FACTOR = float(
    os.getenv("PREDICTION_FACTOR", "2")
)

app = Flask('ride-prediction')  # 🚨 初始化 Web 服务

def prepare_features(ride):
    return {
        "trip_distance": ride["trip_distance"]
    }

def predict(features):
    return features["trip_distance"] * PREDICTION_FACTOR

def handle(ride):
    features = prepare_features(ride)
    pred = predict(features)

    return {
        "prediction": pred
    }

def handle_event(event):
    ride = event["ride"]
    ride_id = event["ride_id"]

    res = handle(ride)

    return {
        "ride_id": ride_id,
        "prediction": res["prediction"]
    }

def handle_encoded_event(encoded_data):
    decoded_data = base64.b64decode(encoded_data).decode("utf-8")
    event = json.loads(decoded_data)

    return handle_event(event)

# 🚨 新增：Web 接口路由，接收 POST 请求
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    event = request.get_json()
    
    # 如果你请求里带了 base64 数据，也可以走 handle_encoded_event
    # 这里为了方便测试，直接解析 JSON 并预测
    ride = event.get("ride")
    ride_id = event.get("ride_id", "test_id")
    
    result = handle_event({"ride": ride, "ride_id": ride_id})
    return jsonify(result)

# 🚨 保留 Lambda 处理逻辑（如果你的课程要求有这个函数的话）
def lambda_handler(event, context):
    predictions = []
    for record in event["Records"]:
        encoded_data = record["kinesis"]["data"]
        pred = handle_encoded_event(encoded_data)
        predictions.append(pred)
    return {"predictions": predictions}

# 🚨 核心修改：启动 Web 服务器，而不是运行一次就退出
if __name__ == "__main__":
    # host 必须是 0.0.0.0，端口 8000
    app.run(host='0.0.0.0', port=8000)