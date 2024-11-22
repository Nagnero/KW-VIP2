import pandas as pd
import numpy as np
import yfinance as yf
from stockstats import StockDataFrame as Sdf
from fredapi import Fred
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import shap

# FRED API 키 설정
fred = Fred(api_key='c7fc62c3e1c6d4001bc7c05a44920857')

# FRED 지표 ID 목록
fred_ID = ["RECPROUSM156N", "CORESTICKM159SFRBATL", "PCETRIM12M159SFRBDAL", "CPALTT01USM657N", "PSAVERT", "AISRSA",
           "ANFCI", "UNEMPLOY"]

# FRED 데이터 가져오기
fred_data = {}
for ID in fred_ID:
    df = fred.get_series(ID, '2002-01-02')
    fred_data[ID] = df

fred_data = pd.DataFrame(fred_data)

# 날짜 인덱스를 사용하여 결측값 보간
fred_data.index = pd.to_datetime(fred_data.index)
fred_data = fred_data.resample('D').mean().ffill()



# 종목 지정 티커
# ticker = '005930.KS'    # 삼성전자
ticker = '036570.KQ'    # 엔씨소프트
# ticker = '105560.KQ'    # KB금융

data = yf.download(ticker, start='2002-02-01', end='2023-12-31')

# 필요한 feature들 추출
data_features = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# StockDataFrame 변환
stock = Sdf.retype(data_features.copy())

# 기술적 지표 추가
# data_features['sma_5'] = stock['close_5_sma']
# data_features['sma_20'] = stock['close_20_sma']
# data_features['sma_40'] = stock['close_40_sma']
# data_features['ema_5'] = stock['close_5_ema']
# data_features['ema_20'] = stock['close_20_ema']
# data_features['ema_40'] = stock['close_40_ema']
# data_features['mstd_5'] = stock['close_5_mstd']
# data_features['mvar_5'] = stock['close_5_mvar']
# data_features['rsv_5'] = stock['close_5_rsv']
# data_features['rsi_14'] = stock['rsi_14']
# data_features['kdjk'] = stock['kdjk']
# data_features['boll'] = stock['boll']
# data_features['macd'] = stock['macd']
# data_features['cr'] = stock['cr']
# data_features['cr-ma2'] = stock['cr-ma2']
# data_features['cr-ma3'] = stock['cr-ma3']
# data_features['wr_14'] = stock['wr_14']
# data_features['cci_14'] = stock['cci_14']
# data_features['mfi_5'] = stock['mfi_5']
# data_features['mfi_14'] = stock['mfi_14']
# data_features['tr'] = stock['tr']
# data_features['atr'] = stock['atr']
# data_features['change_5_kama_5_30'] = stock['close_5,5,30_kama']
# data_features['ndi'] = stock['ndi']
# data_features['adxr'] = stock['adxr']
# data_features['dma'] = stock['dma']
# data_features['dx_14'] = stock['dx_14']
# data_features['trix'] = stock['trix']
# data_features['tema'] = stock['tema']
# data_features['vr'] = stock['vr']
# data_features['mfi'] = stock['mfi']
# data_features['vwma'] = stock['vwma']
# data_features['chop'] = stock['chop']
# data_features['ker'] = stock['ker']
# data_features['kama'] = stock['kama']
# data_features['ppo'] = stock['ppo']
# data_features['stochrsi'] = stock['stochrsi']
# data_features['supertrend'] = stock['supertrend']
# data_features['aroon'] = stock['aroon']
# data_features['ao'] = stock['ao']
# data_features['bop'] = stock['bop']
# data_features['mad'] = stock['close_20_mad']  # Mean Absolute Deviation
# data_features['roc'] = stock['close_12_roc']  # Rate of Change
# data_features['coppock'] = stock['coppock']  # Coppock Curve
# data_features['ichimoku_a'] = stock['ichimoku']  # Ichimoku Cloud
# data_features['ichimoku_b'] = stock['ichimoku_7,22,44']
# data_features['cti'] = stock['cti']  # Correlation Trend Indicator
# data_features['lrma'] = stock['close_10_lrma']  # Linear Regression Moving Average
# data_features['rvgi'] = stock['rvgi']  # Relative Vigor Index
# data_features['rvgis'] = stock['rvgis']
# data_features['rvgi_5'] = stock['rvgi_5']
# data_features['rvgis_5'] = stock['rvgis_5']
# data_features['eribull'] = stock['eribull']  # Elder-Ray Index
# data_features['eribear'] = stock['eribear']
# data_features['eribull_5'] = stock['eribull_5']
# data_features['eribear_5'] = stock['eribear_5']
# data_features['ftr'] = stock['ftr']  # Gaussian Fisher Transform Price Reversals indicator
# data_features['ftr_20'] = stock['ftr_20']
# data_features['inertia'] = stock['inertia']  # Inertia Indicator
# data_features['inertia_10'] = stock['inertia_10']
# data_features['kst'] = stock['kst']  # Know Sure Thing
# data_features['pgo'] = stock['pgo']  # Pretty Good Oscillator
# data_features['pgo_10'] = stock['pgo_10']
# data_features['psl'] = stock['psl']  # Psychological Line
# data_features['psl_10'] = stock['psl_10']
# data_features['high_12_psl'] = stock['high_12_psl']
# data_features['pvo'] = stock['pvo']  # Percentage Volume Oscillator
# data_features['pvos'] = stock['pvos']
# data_features['pvoh'] = stock['pvoh']
# data_features['qqe'] = stock['qqe']
# data_features['qqel'] = stock['qqel']
# data_features['qqes'] = stock['qqes']
# data_features['qqe_10,4'] = stock['qqe_10,4']
# data_features['qqel_10,4'] = stock['qqel_10,4']
# data_features['qqes_10,4'] = stock['qqes_10,4']
# data_features['qqe'] = stock['qqe']

# data_features.to_csv('original_data.csv', index=False)

# 기술적 지표의 시계열 데이터 추가
T = range(-1, -11, -1)
for t in T:
    data_features[f'close_{t}_s'] = stock[f'close_{t}_s']

# 인덱스를 리셋하여 날짜를 포함한 데이터프레임 생성
data_features = data_features.reset_index()




# FRED 데이터와 결합하기 위해 날짜 인덱스 설정
fred_data = fred_data.reset_index()
fred_data.columns = ['Date'] + fred_ID

# 주식 데이터와 FRED 데이터를 날짜 기준으로 결합
data_features['Date'] = pd.to_datetime(data_features['Date'])
merged_features = pd.merge(data_features, fred_data, on='Date', how='left')

# 결측값 처리: 이전 값으로 채우기
merged_features = merged_features.ffill()

# 결측값이 남아있는지 확인하고, 남아있으면 이를 0으로 채움
merged_features = merged_features.fillna(0)



# 새로운 데이터를 저장할 리스트
new_rows = []

# # 각 행 사이의 평균 값을 계산하여 추가
for i in range(len(merged_features) - 1):
    current_row = merged_features.iloc[i]
    next_row = merged_features.iloc[i + 1]

    # 첫 번째 중간 값 계산 (Date 제외)
    first_avg_row = (3 * current_row.drop(labels=['Date']) + next_row.drop(labels=['Date'])) / 4
    first_avg_row['Date'] = pd.to_datetime(current_row['Date']) + (
                pd.to_datetime(next_row['Date']) - pd.to_datetime(current_row['Date'])) / 4
    new_rows.append(first_avg_row)
#
#     # 두 번째 중간 값 계산 (Date 제외)
#     second_avg_row = (2 * current_row.drop(labels=['Date']) + 2 * next_row.drop(labels=['Date'])) / 4
#     second_avg_row['Date'] = pd.to_datetime(current_row['Date']) + 2 * (
#                 pd.to_datetime(next_row['Date']) - pd.to_datetime(current_row['Date'])) / 4
#     new_rows.append(second_avg_row)
#
#     # 세 번째 중간 값 계산 (Date 제외)
#     third_avg_row = (current_row.drop(labels=['Date']) + 3 * next_row.drop(labels=['Date'])) / 4
#     third_avg_row['Date'] = pd.to_datetime(current_row['Date']) + 3 * (
#                 pd.to_datetime(next_row['Date']) - pd.to_datetime(current_row['Date'])) / 4
#     new_rows.append(third_avg_row)

# 새로운 데이터프레임 생성 및 원래 데이터와 결합
new_rows_df = pd.DataFrame(new_rows)
merged_features = pd.concat([merged_features, new_rows_df]).sort_values(by='Date').reset_index(drop=True)




# 5일 후 평균 종가를 기준으로 레이블 생성
merged_features['Target'] = np.where(merged_features['Close'].shift(-5).rolling(5).mean() > merged_features['Close'], 1, 0)

# 결측값 처리 (다시 한 번)
merged_features = merged_features.ffill().fillna(0)
# merged_features.to_csv('data_augmented.csv', index=False)

# Features와 Target 분리
X = merged_features.drop(columns=['Date', 'Target', 'Close'])  # 'Close'는 예측에 사용되므로 제외
y = merged_features['Target']

# 모델 학습
model = GradientBoostingClassifier()
model.fit(X, y)


# SHAP 값 계산
explainer = shap.Explainer(model, X)
shap_values = explainer(X, check_additivity=False)

# SHAP 각 피처의 평균 절대 SHAP 값 계산 후 상위 30개 피처 선택
mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)
top_30_indices = np.argsort(mean_abs_shap_values)[-30:]
top_30_features = X.columns[top_30_indices]
shap_values_top_30 = shap_values[:, top_30_indices]

# SHAP 결과 시각화
shap.summary_plot(shap_values_top_30, X[top_30_features], plot_type="dot", max_display=30)


# TreeSHAP 값 계산
tree_explainer = shap.TreeExplainer(model, X)
tree_shap_values = tree_explainer(X)

print(tree_shap_values.values.shape)  # SHAP 값의 shape 확인
print(tree_shap_values.base_values.shape)  # base_values의 shape 확인

# 각 피처의 평균 절대 TreeSHAP 값 계산 후 상위 30개 피처 선택
mean_abs_tree_shap_values = np.abs(tree_shap_values.values).mean(axis=0)
top_30_indices_tree = np.argsort(mean_abs_tree_shap_values)[-30:]
top_30_features_tree = X.columns[top_30_indices_tree]
tree_shap_values_top_30 = tree_shap_values[:, top_30_indices_tree]

# tree SHAP 값 시각화
shap.summary_plot(tree_shap_values_top_30, X[top_30_features_tree], plot_type="dot", max_display=30)

