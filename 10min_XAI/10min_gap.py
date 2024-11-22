import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score
import shap

stock_df = pd.read_csv('10mindata.csv')
stock_df = stock_df.dropna()

stock_df = stock_df.drop(['logdate'], axis=1)

x = stock_df.iloc[:, 0:-1].values
y = stock_df.iloc[:, -1].values

# 시계열예측이 아니므로 셔플
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.26,  random_state=0, shuffle=True)

# 최대치는 30% 최소치는 -30%이므로 표준화
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)



# 랜덤포레스트 적용 전 최적의 파라미터 서치
# model = RandomForestClassifier()
# grid_rf = {
#     'n_estimators': [20, 50, 100, 500, 1000],
#     'max_depth': np.arange(1, 15, 1),
#     'min_samples_split': [2, 10, 9],
#     'min_samples_leaf': np.arange(1, 15, 2, dtype=int),
#     'bootstrap': [True, False],
#     'random_state': [1, 2, 30, 42]
# }
# rscv = RandomizedSearchCV(estimator=model, param_distributions=grid_rf, cv=3, n_jobs=-1, verbose=2, n_iter=200)
# rscv_fit = rscv.fit(x_train, y_train)
# best_parameters = rscv_fit.best_params_
# print(best_parameters)

model = RandomForestClassifier(random_state=42, n_estimators=20, min_samples_split=2, min_samples_leaf=3, max_depth=6,
                               bootstrap=True)

model.fit(x_train, y_train)
predict = model.predict(x_test)

confusion = confusion_matrix(y_test, predict)
accuracy = accuracy_score(y_test, predict)
precision = precision_score(y_test, predict,average='macro')
recall = recall_score(y_test, predict,average='macro')
f1 = f1_score(y_test, predict,average='macro')
print(confusion)
print('정확도 {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}'.format(accuracy, precision, recall))


shap.initjs()
tree_explainer = shap.TreeExplainer(model)
tree_shap_values = tree_explainer.shap_values(x_train)
sv = np.array(tree_shap_values)
# y = model.predict(x_train).astype("bool")
# sv_survive = sv[:, y, :]
# sv_die = sv[:, ~y, :]
feature_name = stock_df.columns[:-1]
shap.summary_plot(tree_shap_values[1], x_train.astype("float"), feature_names=feature_name)


explainer = shap.Explainer(model)
shap_values = explainer.shap_values(x_train)
shap.summary_plot(shap_values[1], x_train.astype("float"), feature_names=feature_name)

