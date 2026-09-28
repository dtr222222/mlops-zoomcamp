from __future__ import annotations
import argparse
import mlflow
import pandas as pd
from common.data_processing import read_dataframe, generate_dicts
from prefect import task, flow
import os

# 打印也会显示在在 prefect中
@task(log_prints=True)
def predict(model: LinearRegression, dv: DictVectorizer, df: pd.DataFrame, taxi_type: str, year: int, month: int) -> str:
    out_dir = f'output/{taxi_type}'
    _, dicts = generate_dicts(df, dv)
    y_pred = model.predict(dicts)

    mlflow.log_metric('mean_predictions', y_pred.mean())
    print(f'mean predictions: {y_pred.mean()}')

    result = pd.DataFrame({'prediction': y_pred})
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{year}-{month:02}.parquet')
    result.to_parquet(out_path, index=False)

    return out_path

@flow(retries=3, retry_delay_seconds=2)
def run(taxi_type: str, year: int, month: int) -> None:
    path = f'data/{taxi_type}_tripdata_{year}-{month:02}.parquet'
    
    mlflow.set_tracking_uri(f'sqlite:///mlflow.db')
    mlflow.set_experiment('taxi_duration_prediction')

    # 不再使用 run_id，报错频繁，找不到路径，改成用 model_id。
    model_uri = "models:/m-89cf2bc8adea4f68ae9e961cdfd5d809"
    pipeline = mlflow.sklearn.load_model(model_uri)

    pipeline = mlflow.sklearn.load_model(model_uri)
    dv = pipeline.named_steps['dv']
    model = pipeline.named_steps['model']
    df = read_dataframe(path, taxi_type)

    with mlflow.start_run():
        mlflow.log_param('taxi_type', taxi_type)
        mlflow.log_param('year', year)
        mlflow.log_param('month', month)
        predict(model, dv, df, taxi_type, year, month)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='yellow', help='taxi type you choose to predict')
    parser.add_argument('--year', type=int, default=2021, help='year you choose to predict')
    parser.add_argument('--month', type=int, default=1, help='year you choose to predict')

    args = parser.parse_args()

    run(args.type, args.year, args.month)
    


