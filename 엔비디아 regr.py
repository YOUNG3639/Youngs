import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 생성 및 전처리
data = pd.DataFrame({
    'carbon_emissions': [620.73, 620.73, 620.73, 620.73, 799.5, 799.5, 799.5, 799.5, 1029.76, 1029.76, 1029.76, 1029.76, 
                         1326.33, 1326.33, 1326.33, 1326.33, 1708.32, 1708.32, 1708.32, 1708.32, 2200.31, 2200.31, 2200.31, 2200.31, 
                         2834, 2834, 2834, 2834, 3650, 3650, 3650, 3650],
    'global_employment_ratio': [45.38, 45.38, 45.38, 45.38, 46.28, 46.28, 46.28, 46.28, 47.21, 47.21, 47.21, 47.21, 48.15, 48.15, 48.15, 48.15, 
                                49.12, 49.12, 49.12, 49.12, 50.1, 50.1, 50.1, 50.1, 51.1, 51.1, 51.1, 51.1, 48.9, 48.9, 48.9, 48.9],
    'dividend': [0.002875, 0.002875, 0.002875, 0.0035, 0.0035, 0.0035, 0.0035, 0.00375, 0.00375, 0.00375, 0.00375, 0.004, 0.004, 0.004, 0.004, 0.004,
                 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004],
    'operating_profit': [252, 245, 317, 639, 733, 554, 688, 895, 1073, 1295, 1157, 1058, 294, 358, 571, 927, 990, 976, 651, 1398,
                         1507, 1956, 2444, 2671, 2970, 1868, 499, 601, 1256, 2140, 6800, 10417]
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
