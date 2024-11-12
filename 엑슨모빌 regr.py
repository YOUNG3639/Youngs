import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 생성 및 전처리
data = pd.DataFrame({
    'carbon_emissions': [11700, 11700, 11700, 11700, 11700, 11700, 11700, 11700, 11500, 11500, 11500, 11500, 10900, 10900, 10900, 10900, 
                         10200, 10200, 10200, 10200, 10300, 10300, 10300, 10300, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
    'global_employment_ratio': [59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    'dividend': [0.73, 0.75, 0.75, 0.75, 0.75, 0.77, 0.77, 0.77, 0.77, 0.82, 0.82, 0.82, 0.82, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.88, 0.88, 0.88, 0.88, 0.91, 0.91, 0.91, 0.91, 0.95],
    'operating_profit': [1807, 2471, 3332, 812, 6064, 4314, 5694, 3203, 7444, 6659, 9280, 8336, 4470, 4848, 4953, 6615, -9, -1323, -93, -26300, 3850, 6561, 9820, 11950, 8744, 25127, 25631, 19049, 16962, 11905, 13868, 10897]
})

# 데이터 확인
print(data.head())

# 2. 상관관계 분석
correlations = data.corr()

# 상관관계 히트맵
plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

# 3. 회귀 분석을 통한 가중치 계산
# 특성 변수와 목표 변수 분리
X = data[['carbon_emissions', 'global_employment_ratio', 'dividend']]
y = data['operating_profit']

# 데이터 스케일링 (회귀 계수 비교를 용이하게 하기 위함)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 회귀 모델 학습
model = LinearRegression()
model.fit(X_scaled, y)

# 각 특성의 회귀 계수 확인
coefficients = model.coef_
feature_names = X.columns
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")

# 4. 모델 평가 지표 추가 (R^2, MSE, MAE)
y_pred = model.predict(X_scaled)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)

print("\nModel Evaluation:")
print(f"R^2 Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# 5. 환경, 사회, 지배구조 가중치 계산
# 각 특성을 환경, 사회, 지배구조로 그룹화
environment_features = ['carbon_emissions']
social_features = ['global_employment_ratio']
governance_features = ['dividend']

# 그룹별 가중치 합계 계산
env_weight = sum([coef for feature, coef in zip(feature_names, coefficients) if feature in environment_features])
soc_weight = sum([coef for feature, coef in zip(feature_names, coefficients) if feature in social_features])
gov_weight = sum([coef for feature, coef in zip(feature_names, coefficients) if feature in governance_features])

# 총합으로 정규화하여 환경, 사회, 지배구조 가중치를 비율로 계산
total_weight = abs(env_weight) + abs(soc_weight) + abs(gov_weight)
env_weight_ratio = abs(env_weight) / total_weight
soc_weight_ratio = abs(soc_weight) / total_weight
gov_weight_ratio = abs(gov_weight) / total_weight

print("\nCalculated Weights:")
print(f"Environment Weight: {env_weight_ratio:.4f}")
print(f"Social Weight: {soc_weight_ratio:.4f}")
print(f"Governance Weight: {gov_weight_ratio:.4f}")
