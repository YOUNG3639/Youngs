import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 생성 및 전처리
data = pd.DataFrame({
    'carbon_emissions': [6897, 6897, 6897, 6897, 8590, 8590, 8590, 8590, 10775, 10775, 10775, 10775, 11132, 11132, 11132, 11132,
                         12532, 12532, 12532, 12532, 14494, 14494, 14494, 14494, 14923, 14923, 14923, 14923, 14923, 14923, 14923, 14923],
    'global_employment_ratio': [69.8, 69.8, 69.8, 69.8, 69.9, 69.9, 69.9, 69.9, 67.8, 67.8, 67.8, 67.8, 64.5, 64.5, 64.5, 64.5, 
                                60.3, 60.3, 60.3, 60.3, 58.3, 58.3, 58.3, 58.3, 56.4, 56.4, 56.4, 56.4, 54.4, 54.4, 54.4, 54.4],
    'dividend': [400, 400, 20, 550, 140, 140, 140, 430, 354, 354, 354, 354, 354, 354, 354, 354, 354, 354, 354, 1932, 361, 361, 361, 361,
                 361, 361, 361, 361, 361, 361, 361, 361],
    'operating_profit': [6.68, 8.14, 5.2, 9.22, 9.9, 14.07, 14.53, 15.15, 15.64, 14.87, 17.57, 10.8, 6.23, 6.59, 7.78, 7.16,
                         6.45, 8.15, 12.35, 9.05, 9.38, 12.57, 15.82, 13.87, 14.12, 14.1, 10.85, 4.31, 0.64, 0.67, 2.43, 2.82]
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

# 4. 환경, 사회, 지배구조 가중치 계산
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
