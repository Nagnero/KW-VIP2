import pandas as pd
import numpy as np
import yfinance as yf
from stockstats import StockDataFrame as Sdf
from sklearn.ensemble import RandomForestClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt


def get_technical_indicators(stock_data):
    # 필요한 feature들 추출
    return_data_features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    return_data_features.to_csv("엔씨소프트.csv")

    # StockDataFrame 변환
    stock_data_frame = Sdf.retype(return_data_features.copy())

    return_data_features['sma_5'] = stock_data_frame['close_5_sma']
    return_data_features['sma_20'] = stock_data_frame['close_20_sma']
    return_data_features['sma_40'] = stock_data_frame['close_40_sma']
    return_data_features['ema_5'] = stock_data_frame['close_5_ema']
    return_data_features['ema_20'] = stock_data_frame['close_20_ema']
    return_data_features['ema_40'] = stock_data_frame['close_40_ema']
    return_data_features['mstd_5'] = stock_data_frame['close_5_mstd']
    return_data_features['mvar_5'] = stock_data_frame['close_5_mvar']
    return_data_features['rsv_5'] = stock_data_frame['close_5_rsv']
    return_data_features['rsi_14'] = stock_data_frame['rsi_14']
    return_data_features['kdjk'] = stock_data_frame['kdjk']
    return_data_features['boll'] = stock_data_frame['boll']
    return_data_features['macd'] = stock_data_frame['macd']
    return_data_features['cr'] = stock_data_frame['cr']
    return_data_features['cr-ma2'] = stock_data_frame['cr-ma2']
    return_data_features['cr-ma3'] = stock_data_frame['cr-ma3']
    return_data_features['wr_14'] = stock_data_frame['wr_14']
    return_data_features['cci_14'] = stock_data_frame['cci_14']
    return_data_features['mfi_5'] = stock_data_frame['mfi_5']
    return_data_features['mfi_14'] = stock_data_frame['mfi_14']
    return_data_features['tr'] = stock_data_frame['tr']
    return_data_features['atr'] = stock_data_frame['atr']
    return_data_features['change_5_kama_5_30'] = stock_data_frame['close_5,5,30_kama']
    return_data_features['ndi'] = stock_data_frame['ndi']
    return_data_features['adxr'] = stock_data_frame['adxr']
    return_data_features['dma'] = stock_data_frame['dma']
    return_data_features['dx_14'] = stock_data_frame['dx_14']
    return_data_features['trix'] = stock_data_frame['trix']
    return_data_features['tema'] = stock_data_frame['tema']
    return_data_features['vr'] = stock_data_frame['vr']
    return_data_features['mfi'] = stock_data_frame['mfi']
    return_data_features['vwma'] = stock_data_frame['vwma']
    return_data_features['chop'] = stock_data_frame['chop']
    return_data_features['ker'] = stock_data_frame['ker']
    return_data_features['kama'] = stock_data_frame['kama']
    return_data_features['ppo'] = stock_data_frame['ppo']
    return_data_features['stochrsi'] = stock_data_frame['stochrsi']
    return_data_features['supertrend'] = stock_data_frame['supertrend']
    return_data_features['aroon'] = stock_data_frame['aroon']
    return_data_features['ao'] = stock_data_frame['ao']
    return_data_features['bop'] = stock_data_frame['bop']
    return_data_features['mad'] = stock_data_frame['close_20_mad']  # Mean Absolute Deviation
    return_data_features['roc'] = stock_data_frame['close_12_roc']  # Rate of Change
    return_data_features['coppock'] = stock_data_frame['coppock']  # Coppock Curve
    return_data_features['ichimoku_a'] = stock_data_frame['ichimoku']  # Ichimoku Cloud
    return_data_features['ichimoku_b'] = stock_data_frame['ichimoku_7,22,44']
    return_data_features['cti'] = stock_data_frame['cti']  # Correlation Trend Indicator
    return_data_features['lrma'] = stock_data_frame['close_10_lrma']  # Linear Regression Moving Average
    return_data_features['rvgi'] = stock_data_frame['rvgi']  # Relative Vigor Index
    return_data_features['rvgis'] = stock_data_frame['rvgis']
    return_data_features['rvgi_5'] = stock_data_frame['rvgi_5']
    return_data_features['rvgis_5'] = stock_data_frame['rvgis_5']
    return_data_features['eribull'] = stock_data_frame['eribull']  # Elder-Ray Index
    return_data_features['eribear'] = stock_data_frame['eribear']
    return_data_features['eribull_5'] = stock_data_frame['eribull_5']
    return_data_features['eribear_5'] = stock_data_frame['eribear_5']
    return_data_features['ftr'] = stock_data_frame['ftr']  # Gaussian Fisher Transform Price Reversals indicator
    return_data_features['ftr_20'] = stock_data_frame['ftr_20']
    return_data_features['inertia'] = stock_data_frame['inertia']  # Inertia Indicator
    return_data_features['inertia_10'] = stock_data_frame['inertia_10']
    return_data_features['kst'] = stock_data_frame['kst']  # Know Sure Thing
    return_data_features['pgo'] = stock_data_frame['pgo']  # Pretty Good Oscillator
    return_data_features['pgo_10'] = stock_data_frame['pgo_10']
    return_data_features['psl'] = stock_data_frame['psl']  # Psychological Line
    return_data_features['psl_10'] = stock_data_frame['psl_10']
    return_data_features['high_12_psl'] = stock_data_frame['high_12_psl']
    return_data_features['pvo'] = stock_data_frame['pvo']  # Percentage Volume Oscillator
    return_data_features['pvos'] = stock_data_frame['pvos']
    return_data_features['pvoh'] = stock_data_frame['pvoh']
    return_data_features['qqe'] = stock_data_frame['qqe']
    return_data_features['qqel'] = stock_data_frame['qqel']
    return_data_features['qqes'] = stock_data_frame['qqes']
    return_data_features['qqe_10,4'] = stock_data_frame['qqe_10,4']
    return_data_features['qqel_10,4'] = stock_data_frame['qqel_10,4']
    return_data_features['qqes_10,4'] = stock_data_frame['qqes_10,4']
    return_data_features['qqe'] = stock_data_frame['qqe']

    # 기술적 지표의 시계열 데이터 추가
    time = range(-1, -11, -1)
    for t in time:
        return_data_features[f'close_{t}_s'] = stock_data_frame[f'close_{t}_s']

    return return_data_features



# 종목 지정 티커
# ticker = '005930.KS'    # 삼성전자
ticker = '036570.KQ'    # 엔씨소프트
# ticker = '105560.KQ'    # KB금융

# 시계열 데이터 다운로드 후 기술적 지표 추가
data = yf.download(ticker, start='2002-01-02')
data_features = get_technical_indicators(data)

# 인덱스를 리셋하여 날짜를 포함한 데이터프레임 생성
data_features = data_features.reset_index()

# 결측값 처리: 이전 값으로 채우고 0으로 채우기
data_features = data_features.ffill()
data_features = data_features.fillna(0)

# 5일 후 평균 종가를 기준으로 레이블 생성 후 결측값 처리
data_features['Target'] = np.where(data_features['Close'].shift(-5).rolling(5).mean() > data_features['Close'], 1, 0)
merged_features = data_features.ffill().fillna(0)

# Features와 Target 분리
X = merged_features.drop(columns=['Date', 'Target', 'Close'])  # 'Close'는 예측에 사용되므로 제외
y = merged_features['Target']

# 랜덤포레스트 모델 학습
model = RandomForestClassifier()
model.fit(X, y)

# SHAP 값 계산
explainer = shap.Explainer(model, X)
shap_values = explainer(X, check_additivity=False)

print(shap_values.values.shape)

# SHAP 각 피처의 평균 절대 SHAP 값 계산 후 상위 30개 피처 선택
mean_shap_values = shap_values.values.mean(axis=0)
shap_values_label_1 = mean_shap_values[:, 1]
mean_abs_shap_values = np.abs(shap_values_label_1)

top_30_indices = np.argsort(mean_abs_shap_values)[-30:]
top_30_features = X.columns[top_30_indices]

shap_values_array = shap_values.values
shap_values_top_30 = shap_values_array[:, top_30_indices, :]

# SHAP 결과 시각화
# shap.summary_plot(shap_values_top_30, X[top_30_features], plot_type="dot", max_display=30)
shap.summary_plot(shap_values_top_30[:, :, 1], X[top_30_features], plot_type="dot", max_display=30)



# # LIME Explainer 초기화
# lime_explainer = LimeTabularExplainer(
#     X.values,  # 학습 데이터
#     feature_names=X.columns.tolist(),  # 피처 이름
#     class_names=["Decrease", "Increase"],  # 클래스 이름
#     discretize_continuous=False  # 연속형 변수 이산화
# )
#
# # 특정 샘플에 대한 LIME 설명 생성 (예: 첫 번째 샘플)
# sample_idx = 0
# lime_exp = lime_explainer.explain_instance(
#     X.values[sample_idx],
#     model.predict_proba,
#     num_features=30
# )
#
# # LIME 결과 출력
# print(f"LIME Explanation for Sample {sample_idx}")
# for feature, weight in lime_exp.as_list():
#     print(f"{feature}: {weight}")
#
# # 결과 저장
# lime_exp.save_to_file("lime_explanation_sample_0.html")  # HTML 파일 저장
# fig = lime_exp.as_pyplot_figure()  # matplotlib 시각화 생성
# fig.savefig("lime_explanation_sample_0.png")  # PNG 파일로 저장
# plt.close(fig)

