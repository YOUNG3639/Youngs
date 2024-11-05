import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 생성 및 전처리
np.random.seed(42)
data = pd.DataFrame({
    'carbon_emissions': np.random.rand(100),          # 환경 변수: 탄소 배출량
    'energy_efficiency': np.random.rand(100),         # 환경 변수: 에너지 효율성
    'employee_diversity': np.random.rand(100),        # 사회 변수: 직원 다양성 비율
    'labor_conditions': np.random.rand(100),          # 사회 변수: 노동 조건 지표
    'board_diversity': np.random.rand(100),           # 지배구조 변수: 이사회 다양성 비율
    'transparency': np.random.rand(100),              # 지배구조 변수: 투명성 지표
    'financial_performance': np.random.rand(100)      # 목표 변수: 재무 성과 점수
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
X = data[['carbon_emissions', 'energy_efficiency', 'employee_diversity', 'labor_conditions', 'board_diversity', 'transparency']]
y = data['financial_performance']

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
environment_features = ['carbon_emissions', 'energy_efficiency']
social_features = ['employee_diversity', 'labor_conditions']
governance_features = ['board_diversity', 'transparency']

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
