# Q1: Prefect
# Q2: 3.4.25
# Q3: 3403766
# Q4: 3316216

import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('homework')

from prefect import task, flow

@task(retries=3, retry_delay_seconds=2)
def read_dataframe(filename):
    import pyarrow.parquet as pq
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance']

    table = pq.read_table(filename, columns=columns).slice(0, 10000)  # slice(0, N) 表示从第0行开始读N行
    df = table.to_pandas()

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

@task
def create_X(df, dv=None):
    cat = ['PULocationID', 'DOLocationID']
    num = ['trip_distance']
    dicts = df[cat + num].to_dict(orient='records')
    if dv is not None:
        X = dv.transform(dicts)
    else:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)

    return X, dv

@task
def train_model(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

@flow(log_prints=True)
def run(path):
    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        df = read_dataframe(path)
        X_train, dv = create_X(df)
        y_train = df['duration']
        lr = train_model(X_train, y_train)
        y_pred = lr.predict(X_train)
        rmse = root_mean_squared_error(y_true=y_train, y_pred=y_pred)
        print(f'rmse: {rmse:.2f}')
        print(f'intercept: {lr.intercept_:.2f}')

        with open ('models/preprocessor.b', 'wb') as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact('models/preprocessor.b', 'preprocessor')

        mlflow.sklearn.log_model(lr, 'linear_regression_model')
        return run.info.run_id

if __name__ == "__main__":
    path = '/home/dongenxi/mlops-zoomcamp/data/Unit3/yellow_tripdata_2023-03.parquet'
    run_id = run(path)
    uri = f'runs:/{run_id}/model'
    mlflow.register_model(model_uri=uri, name='hw03')