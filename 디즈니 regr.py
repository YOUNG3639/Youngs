import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 생성 및 전처리
data = pd.DataFrame({
    'carbon_emissions': [1800, 1800, 1800, 1800, 1100, 1100, 1100, 1100, 900, 900, 900, 900, 
                         850, 850, 850, 850, 820, 820, 820, 820, 681, 681, 681, 681, 565, 565, 565, 565, 469, 469, 469, 469],
    'global_employment_ratio': [41.7, 41.7, 41.7, 41.7, 42.56, 42.56, 42.56, 42.56, 43.37, 43.37, 43.37, 43.37, 
                                44.2, 44.2, 44.2, 44.2, 45.1, 45.1, 45.1, 45.1, 46, 46, 46, 46, 46.7, 46.7, 46.7, 46.7, 46.1, 46.1, 46.1, 46.1],
    'dividend': [0.66, 0.71, 0.71, 0.78, 0.78, 0.78, 0.78, 0.84, 0.84, 0.84, 0.84, 0.88, 0.88, 0.88, 0.88, 0.88, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3],
    'operating_profit': [3908, 3363, 4101, 2830, 3706, 3750, 3788, 2531, 3778, 4024, 3924, 3078, 
                         3418, 2726, 2544, 2960, 2685, 1216, -4996, -846, 146, 1032, 1320, 507, 
                         2196, 1405, 2390, 542, 1924, 2123, -9, 1062]
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
