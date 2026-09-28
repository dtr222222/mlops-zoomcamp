import base64
import json
import os

PREDICTION_FACTOR = float(
    os.getenv("PREDICTION_FACTOR", "2")
)

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

def lambda_handler(event, context):
    predictions = []

    for record in event["Records"]:
        encoded_data = record["kinesis"]["data"]
        pred = handle_encoded_event(encoded_data)

        predictions.append(pred)

    return {
        "predictions": predictions 
    }





if __name__ == "__main__":
    ride = {
        "trip_distance": 3.66
    }
    features = prepare_features(ride)
    pred = predict(features)

    print(features)
    print(pred)