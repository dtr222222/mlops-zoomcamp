from __future__ import annotations
import pandas as pd
from typing import Optional
import pyarrow.parquet as pq
from sklearn.feature_extraction import DictVectorizer

'''
公用代码。
'''

def read_dataframe(filename: str, taxi_type: str = 'yellow') -> pd.DataFrame:
    '''
    Load the data (Include simple feature engineering).
    '''
    # 虚拟机内存太小，不得已做出的取舍，只读了部分数据。
    table = pq.read_table(filename).slice(0, 1e5)
    df = table.to_pandas()

    # 注意 yellow与 green的某些特征名字不相同，因此要做条件判断。
    if taxi_type == 'yellow':
        df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    elif taxi_type == 'green':
        df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    else:
        # 输入的出租车类型不存在，抛出错误。
        raise ValueError(f'Unknown taxi_type: {taxi_type}')

    df['duration'] = df['duration'].apply(lambda td: td.total_seconds() / 60)
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df

def generate_dicts(df: pd.DataFrame, dv: Optional[DictVectorizer] = None) -> tuple[DictVectorizer, scipy.sparse.csr_matrix]:
    '''
    Through the DictVectorizer generate dicts.
    '''

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    if dv is None:
        dv = DictVectorizer()
        dicts = dv.fit_transform(dicts)
    else:
        dicts = dv.transform(dicts)

    return dv, dicts