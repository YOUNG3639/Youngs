import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. 데이터 생성 (예시용 랜덤 데이터)
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
print("데이터 샘플:")
print(data.head())

# 2. 데이터 분리
X = data[['carbon_emissions', 'energy_efficiency', 'employee_diversity', 'labor_conditions', 'board_diversity', 'transparency']]
y = data['financial_performance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 랜덤 포레스트 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. 모델 예측 및 성능 평가
y_pred = model.predict(X_test)
print("\nMean Squared Error (MSE):", mean_squared_error(y_test, y_pred))

# 5. 특성 중요도 추출
importances = model.feature_importances_
feature_importance = dict(zip(X.columns, importances))
print("\nFeature Importances:")
for feature, importance in feature_importance.items():
    print(f"{feature}: {importance:.4f}")

# 각 특성의 중요도 계산 결과
importances = model.feature_importances_
feature_importance = dict(zip(X.columns, importances))

# 환경, 사회, 지배구조 특성 그룹화
environment_features = ['carbon_emissions', 'energy_efficiency']
social_features = ['employee_diversity', 'labor_conditions']
governance_features = ['board_diversity', 'transparency']

# 그룹별 가중치 합계 계산
env_weight = sum([feature_importance[feature] for feature in environment_features])
soc_weight = sum([feature_importance[feature] for feature in social_features])
gov_weight = sum([feature_importance[feature] for feature in governance_features])

# 가중치 정규화 (비율로 계산)
total_weight = env_weight + soc_weight + gov_weight
env_weight_ratio = env_weight / total_weight
soc_weight_ratio = soc_weight / total_weight
gov_weight_ratio = gov_weight / total_weight

# 결과 출력
print("\nCalculated ESG Weights based on Feature Importances:")
print(f"Environment Weight: {env_weight_ratio:.4f}")
print(f"Social Weight: {soc_weight_ratio:.4f}")
print(f"Governance Weight: {gov_weight_ratio:.4f}")
