import requests

data = {
    'PULocationID': 10,
    'DOLocationID': 50,
    'trip_distance': 40
}

response = requests.post(url=' http://127.0.0.1:9696/predict', json=data)
print(response.json())