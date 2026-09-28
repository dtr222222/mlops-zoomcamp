from pathlib import Path
import argparse
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from common.data_processing import read_dataframe, generate_dicts
import mlflow
from prefect import flow, task


@task
def train(df: pd.DataFrame) -> tuple[DictVectorizer, LinearRegression]:
    '''
    Train the model.
    '''
    dv, dicts = generate_dicts(df)
    y_true = df['duration']

    model = LinearRegression()
    model.fit(dicts, y_true)

    return dv, model

@task
def predict(dv: DictVectorizer, model:LinearRegression, df_val: pd.DataFrame) -> float:
    y_true = df_val['duration']

    _, dicts = generate_dicts(df_val, dv)
    y_pred = model.predict(dicts)

    rmse = root_mean_squared_error(y_true, y_pred)
    print(f'rmse: {rmse}')
    return rmse

@flow(retries=3, retry_delay_seconds=2, log_prints=True)
def run(taxi_type: str, year: int, month: int) -> None:
    '''
    Val dataset is the next month of train dataset.
    '''
    next_year = year + 1 if month == 12 else year
    next_month = month + 1 if month < 12 else 1

    # 以当前目录来执行，设置路径。
    train_path = f'data/{taxi_type}_tripdata_{year}-{month:02}.parquet'
    val_path = f'data/{taxi_type}_tripdata_{next_year}-{next_month:02}.parquet'

    df_train = read_dataframe(train_path, taxi_type)
    df_val = read_dataframe(val_path, taxi_type)

    mlflow.set_tracking_uri(f'sqlite:///mlflow.db')
    mlflow.set_experiment('taxi_duration_prediction')

    # 不要自动登记模型，会影响后面的 log_model。
    # 可能会重复保存模型。
    mlflow.sklearn.autolog(log_models=False)
    with mlflow.start_run():
        mlflow.log_param('taxi_type', taxi_type)
        mlflow.log_param('year', year)
        mlflow.log_param('month', month)
        mlflow.log_param('next_year', next_year)
        mlflow.log_param('next_month', next_month)

        dv, model = train(df_train)
        rmse = predict(dv, model, df_val)
        mlflow.log_metric('rmse', rmse)

        pipeline = Pipeline([('dv', dv), ('model', model)])
        mlflow.sklearn.log_model(pipeline, 'model')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='yellow', help='taxi type you choose to train')
    parser.add_argument('--year', type=int, default=2021, help='year you choose to train')
    parser.add_argument('--month', type=int, default=1, help='month you choose to train')

    args = parser.parse_args()
    run(args.type, args.year, args.month)
