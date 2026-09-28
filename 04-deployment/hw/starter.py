from __future__ import annotations
import pickle
import argparse
import os
import pandas as pd
import mlflow
from prefect import flow, task

@task(retries=3, retry_delay_seconds=2)
def read_data(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

@task
def predict(df: pd.DataFrame, 
            dv: sklearn.feature_extraction._dict_vectorizer.DictVectorizer, 
            model: sklearn.linear_model._base.LinearRegression) -> numpy.ndarray:
    categorical = ['PULocationID', 'DOLocationID']

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    return y_pred

@task
def save_result(df: pd.DataFrame, output_file: str, preds: numpy.ndarray) -> None:
    df_result = pd.DataFrame()

    df_result['ride_id'] = df['ride_id']
    df_result['predict_duration'] = preds

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

@flow(retries=3, retry_delay_seconds=2)
def run(year: int, month: int) -> None: 
    mlflow.set_tracking_uri('sqlite:////app/mlflow.db')
    mlflow.set_experiment('yellow_taxi_duration_prediction')
    
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02}.parquet')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    output_file = f'output/yellow/{year}-{month}.parquet'

    preds = predict(df, dv, model)
    print(f'mean: {preds.mean()}')

    with mlflow.start_run():
        mlflow.log_param('year', year)
        mlflow.log_param('month', month)

        mlflow.log_param('mean_prediction', preds.mean())
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_result(df, output_file, preds)

        mlflow.log_artifact(output_file, artifact_path='predictions')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True, help='year you choose to predict')
    parser.add_argument('--month', type=int, required=True, help='month you choose to predict')
    args = parser.parse_args()

    run(args.year, args.month)
   

