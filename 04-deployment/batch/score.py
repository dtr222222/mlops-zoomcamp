import pandas as pd
import uuid
import mlflow
import argparse    
from prefect import task, flow

# task里不推荐嵌套 task
def generate_uuids(n: int):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid1()))
    return ride_ids

@task(retries=3, retry_delay_seconds=3)
def read_dataframe(filename: str, taxi_type:str) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    if taxi_type == 'green':
        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    elif taxi_type == 'yellow':
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime

    df['duration'] = df['duration'].dt.total_seconds() / 60
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
    
    return df

@task
def prepare_dicts(df: pd.DataFrame) -> dict:
    cat = ['PULocationID', 'DOLocationID']
    df[cat] = df[cat].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    cat = ['PU_DO']
    num = ['trip_distance']
    dicts = df[cat + num].to_dict(orient='records')

    return dicts

@task
def edit_doc(df: pd.DataFrame, pred) -> pd.DataFrame:
    doc = pd.DataFrame()
    
    doc['ride_id'] = generate_uuids(len(df))
    doc['PULocationID'] = df['PULocationID']
    doc['DOLocationID'] = df['DOLocationID']
    doc['predict_duration'] = pred
    doc['actual_duration'] = df['duration']
    doc['diff'] = doc['actual_duration'] - doc['predict_duration']

    return doc

@flow(retries=3, retry_delay_seconds=2)
def ride_duration_prediction(taxi_type: str, year: int, month: int) -> None:
    

    input_path = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year}-{month:02}.parquet'
    output_path = f'output/{taxi_type}/{year}-{month:02}.parquet'

    df = read_dataframe(input_path, taxi_type)
    dicts = prepare_dicts(df)

    model_uri = 'runs:/2c2a1858070a407283f4e07cc29dcaed/model'
    model = mlflow.pyfunc.load_model(model_uri)

    y_pred = model.predict(dicts)
    doc = edit_doc(df, y_pred)
    doc.to_parquet(path=output_path, index=False)

if __name__ == '__main__':
    # 不要把下面关于命令行参数的代码写到 run中，不然无法通过 prefect的 ui来写参数
    parser = argparse.ArgumentParser(description='predict the duration')
    parser.add_argument('--taxi_type', type=str, required=True, help='choose the taxi type you will predict')
    parser.add_argument('--year', type=int, required=True, help='choose the year you will predict')
    parser.add_argument('--month', type=int, required=True, help='choose the month you will predict')
    args = parser.parse_args()

    ride_duration_prediction(args.taxi_type, args.year, args.month)
