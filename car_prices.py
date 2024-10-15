import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error

from autogluon.tabular import TabularDataset, TabularPredictor

from preprocess import preprocess_data, test_preprocess_data

# 1. 데이터 로드
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

train_df, scaler, lower_bound, upper_bound, categorical_features = preprocess_data(train_df)
test_df = test_preprocess_data(test_df, lower_bound, upper_bound, scaler,categorical_features)

# 2. 타겟 컬럼과 식별자 컬럼 정의
label = 'price'  # 예측하고자 하는 타겟 변수
id_column = 'id'  # 식별자 컬럼

train_data = train_df.drop(columns=[id_column])  # id 컬럼 제외
test_data = test_df.drop(columns=[id_column])  # id 컬럼 제외

# 3. 모델 학습
predictor = TabularPredictor(label=label,eval_metric = 'rmse').fit(train_data=train_data,
                                                                                  presets = 'best_quality',
                                                                                  auto_stack = True,
                                                                                  num_bag_folds = 9,
                                                                                  #num_stack_levels = 1,
                                                                                  time_limit = 3600 * 10,
                                                                                  verbosity = 1)

# 4. 테스트 데이터로 예측
predictions = predictor.predict(test_data)

# 5. submission.csv 파일 생성
submission = test_df[[id_column]].copy()  # id 컬럼만 가져옴
submission[label] = predictions  # 예측 결과를 price 컬럼으로 추가

submission = submission.sort_values(by='id', ascending=True)

# 6. 결과를 submission.csv 파일로 저장
submission.to_csv('submission_0911_1.csv', index=False)

print("Submission file created successfully!")
