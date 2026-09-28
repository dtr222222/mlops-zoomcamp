from app import (
    prepare_features, 
    predict, 
    handle, 
    handle_event,
    handle_encoded_event,
    lambda_handler
)
import json
import base64

def test_prepare_features():
    ride = {
        "trip_distance": 3.66
    }

    features = prepare_features(ride)

    assert features == {
        "trip_distance": 3.66
    }

def test_predict():
    features = {
        "trip_distance": 3.66
    }

    pred = predict(features)

    assert pred == 7.32

def test_handle():
    ride = {
        "trip_distance": 3.66
    }

    res = handle(ride)

    assert res == {
        "prediction": 7.32
    }

def test_handle_event():
    event = {
        "ride": {
            "trip_distance": 3.66
        },
        "ride_id": 256
    }

    res = handle_event(event)

    assert res == {
        "ride_id": 256,
        "prediction": 7.32
    }

def test_handle_encoded_event():
    event = {
        "ride": {
            "trip_distance": 3.66
        },
        "ride_id": 256
    }

    json_data = json.dumps(event)
    encoded_data = base64.b64encode(json_data.encode("utf-8")).decode("utf-8")

    res = handle_encoded_event(encoded_data)

    assert res == {
        "ride_id": 256,
        "prediction": 7.32
    }

def test_lambda_handler():
    events = [
        {
            "ride": {
                "trip_distance": 3.66
            },
            "ride_id": 256
        },
        {
            "ride": {
                "trip_distance": 5.0
            },
            "ride_id": 257
        }
    ]

    records = []

    for event in events:
        json_data = json.dumps(event)
        encoded_data = base64.b64encode(json_data.encode("utf-8")).decode("utf-8")

        records.append({
            "kinesis":{
                "data": encoded_data
            }
        })

    res = lambda_handler(
        {"Records": records},
        None
    )

    assert res == {
        "predictions": [
            {
                "ride_id": 256,
                "prediction": 7.32
            },
            {
                "ride_id": 257,
                "prediction": 10.0
            }
        ]
    }
