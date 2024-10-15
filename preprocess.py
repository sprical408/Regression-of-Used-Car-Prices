import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# 전처리 Ver 2
# 조건에 따라 fuel_type의 NaN 값을 대체
def fill_fuel_type(row):
    if pd.isna(row['fuel_type']):
        if 'electric' in row['engine'].lower() or 'kW' in row['engine'].lower() or 'Battery' in row['engine'].lower() or 'Dual Motor - Standard' in row['engine'].lower():
            return 'Electric'
        elif 'Hybrid' in row['engine'].lower():
            return 'Hybrid'
        elif 'Flex' in row['engine'].lower():
            return 'E85 Flex Fuel'
        elif 'Diesel' in row['engine'].lower():
            return 'Diesel'
        else:
            return 'Gasoline'
    else:
        return row['fuel_type']

# 결측치 대체 함수 정의
def fill_missing_values(df):
    # 1. Color와 Fuel_type을 Model 기준으로 최빈값으로 채우기
    for column in ['ext_col', 'int_col', 'fuel_type']:
        df[column] = df.groupby('model')[column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))

    # 2. Engine을 Fuel_type 기준으로 최빈값으로 채우기
    df['engine'] = df.groupby('fuel_type')['engine'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))

    # 3. Transmission을 Engine 기준으로 최빈값으로 채우기
    df['transmission'] = df.groupby('engine')['transmission'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))

    return df

def Scaler(df):
    # milage 데이터프레임이 포함된 경우 예시
    Q1 = df['milage'].quantile(0.25)
    Q3 = df['milage'].quantile(0.75)
    IQR = Q3 - Q1

    # IQR에 1.5를 곱해 이상치 경계 설정
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 이상치를 상한선 또는 하한선으로 대체
    df['milage'] = df['milage'].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))

    # Min-Max 정규화 적용
    scaler = MinMaxScaler()

    # milage 열을 2D 배열로 변환한 후 정규화 적용, 그 결과를 다시 milage 열로 대체
    df['milage'] = scaler.fit_transform(df[['milage']])

    return df, scaler ,lower_bound, upper_bound

def preprocess_data(df):
    # 결측치 처리

    # 범주형 변수의 열 인덱스 또는 이름을 식별합니다
    #categorical_features = [col for i, col in enumerate(df.columns) if df[col].dtype == 'object']
    categorical_features = ['brand', 'model','model_year', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col','accident','clean_title']
    df.loc[(df['brand'] == 'Tesla') & (df['fuel_type'].isnull()), 'fuel_type'] = 'Electric'
    df['fuel_type'].replace(['not supported', '–'], np.nan, inplace=True)
    # apply 함수를 사용해 각 행에 대해 fill_fuel_type 함수를 적용
    df['fuel_type'] = df.apply(fill_fuel_type, axis=1)

    df = df.fillna('Unknown')
    # '-'를 결측값으로 처리
    df = df.replace('-', np.nan)

    df = fill_missing_values(df)

    #df.replace('–', pd.NA, inplace=True)
    #df.fillna('Unknown', inplace=True)

    df, scaler, lower_bound, upper_bound = Scaler(df)

    for col in categorical_features:
        df[col] = df[col].astype('category')

    return df, scaler, lower_bound, upper_bound, categorical_features

def test_preprocess_data(df, lower_bound, upper_bound, scaler, categorical_features):

    df.loc[(df['brand'] == 'Tesla') & (df['fuel_type'].isnull()), 'fuel_type'] = 'Electric'
    df['fuel_type'].replace(['not supported', '–'], np.nan, inplace=True)
    df['fuel_type'] = df.apply(fill_fuel_type, axis=1)

    df = df.fillna('Unknown')
    # '-'를 결측값으로 처리
    df = df.replace('-', np.nan)

    df = fill_missing_values(df)

    # df.replace('–', pd.NA, inplace=True)
    # df.fillna('Unknown', inplace=True)

    # 이상치를 상한선 또는 하한선으로 대체
    df['milage'] = df['milage'].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))

    # milage 열을 2D 배열로 변환한 후 정규화 적용, 그 결과를 다시 milage 열로 대체
    df['milage'] = scaler.fit_transform(df[['milage']])

    for col in categorical_features:
        df[col] = df[col].astype('category')

    return df