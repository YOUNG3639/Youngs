# Python IDLE 3.12 64-bit, CPU 환경에서 실행하였을 때 아래의 결과가 도출됨
# 다른 에디터(Google Colab)를 활용하거나 Python IDLE의 버전이 상이하거나 GPU 환경에서 실행할 경우, 결과값에 차이가 발생할 가능성이 높음
# 또한 활용한 라이브러리의 추후 업데이트로 인해 결과값에 차이가 발생할 가능성이 존재함

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 시드 고정
random.seed(77)
np.random.seed(77)
torch.manual_seed(77)

# 재현성을 위한 추가 설정
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False  

# 인플레이션율
inflation_rate = np.array([
    7.643422, 8.755917, 8.986059, 8.877884, 9.741305, 8.856339, 9.323949, 9.419388, 
    7.152639, 7.005698, 6.028852, 4.737355, 4.625847, 4.678690, 4.434054, 5.458738, 
    6.519653, 5.837591, 6.912941, 5.800605, 4.758882, 4.826918, 3.903785, 4.449258, 
    4.673664, 4.874383, 5.065516, 5.077813, 4.692810, 4.041592, 3.968658, 5.051511, 
    8.934591, 8.171770, 7.031072, 5.973699, 0.708999, 0.594169, 0.651854, 1.296209, 
    1.923421, 1.586993, 2.981707, 2.539978, 3.694487, 5.042809, 4.246131, 3.297529, 
    2.524460, 2.688524, 2.514363, 3.316559, 4.075324, 3.305247, 3.168001, 3.522594, 
    3.246781, 3.410215, 4.331604, 3.370134, 3.250806, 2.977544, 2.336224, 2.462332, 
    2.081389, 2.238585, 2.523900, 2.124841, 2.003180, 2.448947, 2.299826, 3.379060, 
    3.801694, 4.845073, 5.541796, 4.495450, 3.911460, 2.765350, 1.980090, 2.407549, 
    2.983924, 2.663502, 2.873918, 3.234479, 3.838733, 3.981643, 4.315543, 3.963747, 
    3.012184, 2.429132, 1.610372, 1.715957, 1.557701, 1.215215, 1.366212, 1.068619, 
    1.128022, 1.608019, 1.381064, 0.982181, 0.697150, 0.563234, 0.645667, 0.920542, 
    0.859326, 0.840646, 0.735124, 1.451428, 2.199702, 1.921448, 2.218510, 1.441045, 
    1.076691, 1.492170, 1.546318, 1.786975, 0.541741, 0.652200, 0.045261, 0.295617, 
    0.963624, 0.009042, 0.724849, 0.454149, 1.429143, 2.491210, 2.535183, 3.542914, 
    3.911715, 5.370969, 5.840553, 5.214458
])

# 명목 금리
interest_rate = np.array([
    7.000, 7.000, 7.000, 7.000, 7.000, 7.000, 7.000, 7.000, 7.000, 7.000, 7.000, 7.000, 
    5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 
    5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 3.000, 3.000, 
    3.000, 3.000, 3.000, 3.000, 5.000, 5.000, 5.000, 5.250, 5.000, 5.000, 4.000, 4.000, 
    4.000, 4.250, 4.250, 4.250, 4.250, 4.000, 3.750, 3.750, 3.750, 3.750, 3.500, 3.250, 
    3.250, 3.250, 3.250, 3.750, 4.000, 4.250, 4.500, 4.500, 4.500, 4.500, 5.000, 5.000, 
    5.000, 5.000, 5.250, 3.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.250, 2.500, 
    3.000, 3.250, 3.250, 3.250, 3.250, 3.250, 3.000, 2.750, 2.750, 2.500, 2.500, 2.500, 
    2.500, 2.500, 2.250, 2.000, 1.750, 1.500, 1.500, 1.500, 1.500, 1.250, 1.250, 1.250, 
    1.250, 1.250, 1.250, 1.500, 1.500, 1.500, 1.500, 1.750, 1.750, 1.750, 1.500, 1.250, 
    0.750, 0.500, 0.500, 0.500, 0.500, 0.500, 0.750, 1.000, 1.250, 1.750, 2.500, 3.250
])

# 환율
exchange_rate = np.array([
    690.560, 710.260, 715.480, 715.010, 721.570, 725.330, 733.010, 753.040, 766.500, 
    783.310, 787.390, 785.410, 793.940, 799.520, 807.840, 808.890, 809.060, 807.260, 
    803.200, 795.560, 786.310, 763.290, 765.710, 769.300, 783.000, 786.100, 816.860, 
    831.510, 865.320, 891.720, 898.240, 1140.920, 1606.120, 1394.520, 1325.210, 1279.860, 
    1198.680, 1191.410, 1194.160, 1174.460, 1125.410, 1116.290, 1115.270, 1164.110, 
    1271.680, 1305.680, 1293.780, 1291.840, 1319.630, 1270.680, 1196.340, 1220.960, 
    1201.110, 1209.810, 1175.750, 1181.250, 1171.850, 1161.920, 1155.210, 1093.970, 
    1022.480, 1007.960, 1029.190, 1037.000, 977.520, 950.410, 955.020, 938.400, 938.900, 
    929.260, 928.170, 920.590, 955.970, 1016.720, 1062.640, 1362.790, 1415.220, 1288.680, 
    1240.890, 1168.610, 1144.080, 1163.460, 1185.590, 1132.770, 1120.400, 1083.890, 
    1083.040, 1144.750, 1131.470, 1151.810, 1133.540, 1090.860, 1084.080, 1122.150, 
    1112.180, 1062.100, 1069.010, 1030.380, 1025.760, 1086.720, 1100.260, 1097.770, 
    1167.800, 1157.690, 1201.440, 1163.300, 1121.370, 1157.370, 1154.280, 1129.430, 
    1132.240, 1105.720, 1072.290, 1078.570, 1121.590, 1127.520, 1125.080, 1165.910, 
    1193.240, 1175.810, 1193.600, 1220.810, 1188.540, 1117.640, 1114.050, 1121.230, 
    1157.350, 1183.170, 1204.950, 1259.570, 1337.980, 1359.260
])

# 단위당 노동비용 증가율
unit_labor_cost_change = np.array([
    5.815204668, 7.074247686, 7.733419482, 6.139760946, 8.129899885, 7.994266516, 
    6.73590414, 6.868350464, 5.183407428, 4.780261299, 3.52475122, 3.313291358, 
    4.223674349, 5.532291198, 6.448852773, 6.287569548, 5.283000145, 5.064948774, 
    5.398221821, 7.77044735, 6.776782309, 7.005604256, 7.453619989, 5.204326987, 
    6.159674284, 5.801021287, 5.465872305, 5.145936839, 3.133265331, 5.175078293, 
    5.034867247, 4.129533963, 1.327762937, -1.312385622, 0.672162506, 3.053926002, 
    9.788373766, 11.12716983, 8.820893113, 7.3376506, 6.42654912, 4.293250104, 
    5.028339057, 3.456048467, 3.701473835, 2.403816291, 1.556487421, 3.239156162, 
    2.642570888, 4.77504266, 5.794590883, 5.647419546, 3.635196698, 2.766868423, 
    2.675091382, 3.678557417, 3.391206037, 4.432261589, 3.254658899, 1.235287669, 
    2.397591175, 3.124603985, 4.32441768, 4.660430876, 4.82327292, 3.43328583, 
    3.293000086, 3.069342611, 3.14359361, 4.236201396, 3.806606521, 5.292854897, 
    4.298306927, 3.146949773, 2.99044487, -1.941235088, -1.312007653, -0.473636701, 
    1.177192788, 5.332497142, 6.655437554, 5.832312243, 4.170206309, 4.526314548, 
    3.113493363, 1.728158023, 1.375311173, 0.236841112, 0.253504483, 0.623528619, 
    0.468277636, 1.224288173, 1.635813815, 1.997847749, 1.902849834, 1.504443355, 
    0.562821405, 1.25909378, 0.645378348, 0.795933051, 1.170231323, 1.058090206, 
    2.316077622, 2.254585264, 2.161847583, 2.836631127, 1.649289548, 1.54805692, 
    1.905732817, 1.332222306, 2.709669184, 1.824333732, 2.432875677, 2.627370602, 
    2.161528317, 2.909420739, 1.24326425, 1.38733405, 0.724934719, 1.094127448, 
    0.356843132, -1.22697332, 0.292834216, 0.916820931, 3.87487437, 4.072705587, 
    2.023163137, 1.741955003, -0.662573955, -0.238915432, 0.246100426, -0.830166125
])

# 실업자 수 증가율
unemployment_change = np.array([
    -1.356081, -8.163662, 1.428216, 0.913788, 3.341082, 2.835107, -1.726078, 1.215580, 
    2.639810, 7.843239, 8.301515, 7.542992, 10.919631, 25.291730, 16.971441, 14.747640, 
    -0.275533, -14.843739, -14.208988, -19.170340, -18.722690, -18.748541, -11.039293, 
    -7.478707, -6.141786, 0.977200, -1.648930, 13.320556, 40.283456, 30.940759, 20.178766, 
    27.439123, 83.567471, 171.191655, 238.971148, 181.623280, 47.286355, 0.870921, -20.091448, 
    -34.508112, -32.378139, -36.238546, -32.103366, -18.824546, -4.237374, -7.283092, 
    -11.142690, -11.098094, -20.804014, -15.238163, -13.858728, -14.371831, -4.925436, 
    8.153150, 16.345057, 22.994064, 9.920696, 4.611435, 3.474767, 0.977988, 5.638839, 
    4.486295, 3.030645, -1.310117, -5.246796, -6.979093, -6.370785, -4.039559, -6.386135, 
    -3.365353, -5.687342, -5.850642, -6.086135, -3.624041, -1.142451, 3.944940, 12.915576, 
    22.061041, 17.851252, 8.837940, 23.174957, -8.948464, -0.815377, -0.468907, -9.351670, 
    -0.275807, -9.412230, -7.238732, -8.304432, -3.342151, -1.846966, -2.233065, -4.735084, 
    -4.634488, 0.297324, 1.921249, 12.397134, 19.759977, 14.838591, 17.418398, 5.599507, 
    5.525748, 5.339647, -0.961174, 5.480260, -1.992905, 6.593001, 3.989839, 1.033789, 
    4.029376, -1.357290, 1.455093, 1.798098, 2.581056, 11.761697, 5.453314, 5.051509, 
    5.965855, -11.183794, -5.799745, -6.422689, 4.459359, 6.783311, 16.522254, 17.761551, 
    -7.818783, -18.864482, -20.827733, -28.559146, -21.964566, -10.528277, -11.412121
])







# 년도와 분기
years = []
start_year = 1990
quarters = ['1Q', '2Q', '3Q', '4Q']
for i in range(len(interest_rate)):
    year = start_year + i // 4
    quarter = quarters[i % 4]
    years.append(f'{year}/{quarter}')


# ReplayBuffer 클래스 정의
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# DQN 모델 정의
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 강화 학습 환경 정의
class EconomicEnvironment:
    def __init__(self, inflation_rate, interest_rate, exchange_rate, unit_labor_cost_change, unemployment_change):
        self.inflation_rate = inflation_rate
        self.interest_rate = interest_rate
        self.exchange_rate = exchange_rate
        self.unit_labor_cost_change = unit_labor_cost_change
        self.unemployment_change = unemployment_change

        self.state = [7.643422, 7.000, 690.560, 5.815204668, -1.356081]
        self.max_steps = len(inflation_rate)
        self.current_step = 0
        self.neutral_rate = np.zeros(len(inflation_rate))  # 중립금리 기록
        self.neutral_rate[0] = self.interest_rate[0]  # 중립금리 초기값 설정

    def reset(self):
        self.current_step = 0
        self.state = [self.inflation_rate[0], self.interest_rate[0], self.exchange_rate[0],
                      self.unit_labor_cost_change[0], self.unemployment_change[0]]
        return np.array(self.state)

    def step(self, action):
        t = self.current_step

        # 현재 시점의 경제 상태
        inflation_rate_t = self.inflation_rate[t]
        nominal_rate_t = self.interest_rate[t]
        exchange_rate_t = self.exchange_rate[t]
        unit_labor_cost_change_t = self.unit_labor_cost_change[t]
        unemployment_change_t = self.unemployment_change[t]
        neutral_rate_t = self.neutral_rate[t]

        # 행동에 따른 중립금리 조정
        if action == 0:  # 일반적인 금리 인상
            if 0 < self.inflation_rate[t-1] - self.inflation_rate[t-2] < 2:
                neutral_rate_t += random.uniform(0.1, 0.3)  # 소폭 인상
            elif 2 <= self.inflation_rate[t-1] - self.inflation_rate[t-2] < 3:
                neutral_rate_t += random.uniform(0.3, 0.5)  # 일반적인 인상
            elif self.inflation_rate[t-1] - self.inflation_rate[t-2] >= 3:
                neutral_rate_t += random.uniform(0.5, 1.0)  # 대폭 인상
        elif action == 1:  # 일반적인 금리 하락
            if 0 < self.inflation_rate[t-2] - self.inflation_rate[t-1] < 2:
                neutral_rate_t -= random.uniform(0.1, 0.3)  # 소폭 인하
            elif 2 <= self.inflation_rate[t-2] - self.inflation_rate[t-1] < 3:
                neutral_rate_t -= random.uniform(0.3, 0.5)  # 일반적인 인하
            elif self.inflation_rate[t-2] - self.inflation_rate[t-1] >= 3:
                neutral_rate_t -= random.uniform(0.5, 1.0)  # 대폭 인하
        elif action == 2:  # 공급 충격 시 금리 하락
            if 0 < self.inflation_rate[t-1] - self.inflation_rate[t-2] < 2:
                neutral_rate_t -= random.uniform(0.1, 0.3)  # 소폭 인하
            elif 2 <= self.inflation_rate[t-1] - self.inflation_rate[t-2] < 3:
                neutral_rate_t -= random.uniform(0.3, 0.5)  # 일반적인 인하
            elif self.inflation_rate[t-1] - self.inflation_rate[t-2] >= 3:
                neutral_rate_t -= random.uniform(0.5, 1.0)  # 대폭 인하
        elif action == 3:  # 경기 회복 시 금리 인상
            if 0 < self.inflation_rate[t-2] - self.inflation_rate[t-1] < 2:
                neutral_rate_t += random.uniform(0.1, 0.3)  # 소폭 인상
            elif 2 <= self.inflation_rate[t-2] - self.inflation_rate[t-1] < 3:
                neutral_rate_t += random.uniform(0.3, 0.5)  # 일반적인 인상
            elif self.inflation_rate[t-2] - self.inflation_rate[t-1] >= 3:
                neutral_rate_t += random.uniform(0.5, 1.0)  # 대폭 인상

        # 중립금리 제약 조건 적용 (명목금리의 ±3%p 범위 내)
        lower_bound = nominal_rate_t - 3.0
        upper_bound = nominal_rate_t + 3.0
        neutral_rate_t = min(max(neutral_rate_t, lower_bound), upper_bound)

        self.neutral_rate[t] = neutral_rate_t

        reward = 0
        # t=0,1일 때는 보상 조건 없이 패스
        if t == 0:
            self.current_step += 1
            return np.array([self.inflation_rate[self.current_step], self.interest_rate[self.current_step],
                             self.exchange_rate[self.current_step], self.unit_labor_cost_change[self.current_step],
                             self.unemployment_change[self.current_step]]), reward, False

        if t == 1:
            self.current_step += 1
            return np.array([self.inflation_rate[self.current_step], self.interest_rate[self.current_step],
                             self.exchange_rate[self.current_step], self.unit_labor_cost_change[self.current_step],
                             self.unemployment_change[self.current_step]]), reward, False

        if t >= 2:  # t-2 조건을 위한 체크
            # 1점 조건
            if (self.unemployment_change[t-2] > self.unemployment_change[t-1] and
                self.unit_labor_cost_change[t-2] < self.unit_labor_cost_change[t-1] and
                self.inflation_rate[t-2] < self.inflation_rate[t-1] and
                self.neutral_rate[t-1] < neutral_rate_t):

                if 0 < self.inflation_rate[t-1] - self.inflation_rate[t-2] < 2:
                    neutral_rate_t += random.uniform(0.1, 0.3)  # 소폭 인상
                elif 2 <= self.inflation_rate[t-1] - self.inflation_rate[t-2] < 3:
                    neutral_rate_t += random.uniform(0.3, 0.5)  # 일반적인 인상
                elif self.inflation_rate[t-1] - self.inflation_rate[t-2] >= 3:
                    neutral_rate_t += random.uniform(0.5, 1.0)  # 대폭 인상

                reward += 1

            if (self.unemployment_change[t-2] < self.unemployment_change[t-1] and
                self.unit_labor_cost_change[t-2] > self.unit_labor_cost_change[t-1] and
                self.inflation_rate[t-2] > self.inflation_rate[t-1] and
                self.neutral_rate[t-1] > neutral_rate_t):

                if 0 < self.inflation_rate[t-2] - self.inflation_rate[t-1] < 2:
                    neutral_rate_t -= random.uniform(0.1, 0.3)  # 소폭 인하
                elif 2 <= self.inflation_rate[t-2] - self.inflation_rate[t-1] < 3:
                    neutral_rate_t -= random.uniform(0.3, 0.5)  # 일반적인 인하
                elif self.inflation_rate[t-2] - self.inflation_rate[t-1] >= 3:
                    neutral_rate_t -= random.uniform(0.5, 1.0)  # 대폭 인하

                reward += 1

            if (self.exchange_rate[t-2] < self.exchange_rate[t-1] and
                self.unit_labor_cost_change[t-2] < self.unit_labor_cost_change[t-1] and
                self.inflation_rate[t-2] < self.inflation_rate[t-1] and
                self.neutral_rate[t-1] < neutral_rate_t):
                
                if 0 < self.inflation_rate[t-1] - self.inflation_rate[t-2] < 2:
                    neutral_rate_t += random.uniform(0.1, 0.3)  # 소폭 인상
                elif 2 <= self.inflation_rate[t-1] - self.inflation_rate[t-2] < 3:
                    neutral_rate_t += random.uniform(0.3, 0.5)  # 일반적인 인상
                elif self.inflation_rate[t-1] - self.inflation_rate[t-2] >= 3:
                    neutral_rate_t += random.uniform(0.5, 1.0)  # 대폭 인상

                reward += 1

            if (self.exchange_rate[t-2] > self.exchange_rate[t-1] and
                self.unit_labor_cost_change[t-2] > self.unit_labor_cost_change[t-1] and
                self.inflation_rate[t-2] > self.inflation_rate[t-1] and
                self.neutral_rate[t-1] > neutral_rate_t):

                if 0 < self.inflation_rate[t-2] - self.inflation_rate[t-1] < 2:
                    neutral_rate_t -= random.uniform(0.1, 0.3)  # 소폭 인하
                elif 2 <= self.inflation_rate[t-2] - self.inflation_rate[t-1] < 3:
                    neutral_rate_t -= random.uniform(0.3, 0.5)  # 일반적인 인하
                elif self.inflation_rate[t-2] - self.inflation_rate[t-1] >= 3:
                    neutral_rate_t -= random.uniform(0.5, 1.0)  # 대폭 인하

                reward += 1

            if (self.exchange_rate[t-2] < self.exchange_rate[t-1] and
                self.unit_labor_cost_change[t-2] > self.unit_labor_cost_change[t-1] and
                self.inflation_rate[t-2] < self.inflation_rate[t-1] and
                self.neutral_rate[t-1] > neutral_rate_t):

                if 0 < self.inflation_rate[t-1] - self.inflation_rate[t-2] < 2:
                    neutral_rate_t -= random.uniform(0.1, 0.3)  # 소폭 인하
                elif 2 <= self.inflation_rate[t-1] - self.inflation_rate[t-2] < 3:
                    neutral_rate_t -= random.uniform(0.3, 0.5)  # 일반적인 인하
                elif self.inflation_rate[t-1] - self.inflation_rate[t-2] >= 3:
                    neutral_rate_t -= random.uniform(0.5, 1.0)  # 대폭 인하

                reward += 1

            if (self.exchange_rate[t-2] > self.exchange_rate[t-1] and
                self.unit_labor_cost_change[t-2] < self.unit_labor_cost_change[t-1] and
                self.inflation_rate[t-2] > self.inflation_rate[t-1] and
                self.neutral_rate[t-1] < neutral_rate_t):
                
                if 0 < self.inflation_rate[t-2] - self.inflation_rate[t-1] < 2:
                    neutral_rate_t += random.uniform(0.1, 0.3)  # 소폭 인상
                elif 2 <= self.inflation_rate[t-2] - self.inflation_rate[t-1] < 3:
                    neutral_rate_t += random.uniform(0.3, 0.5)  # 일반적인 인상
                elif self.inflation_rate[t-2] - self.inflation_rate[t-1] >= 3:
                    neutral_rate_t += random.uniform(0.5, 1.0)  # 대폭 인상

                reward += 1

        if t >= 3:  # t-3 조건을 위한 체크
            # 2점 조건
            if (self.unemployment_change[t-3] > self.unemployment_change[t-2] > self.unemployment_change[t-1] and
                self.unit_labor_cost_change[t-3] < self.unit_labor_cost_change[t-2] < self.unit_labor_cost_change[t-1] and
                self.inflation_rate[t-3] < self.inflation_rate[t-2] < self.inflation_rate[t-1] and
                self.neutral_rate[t-1] < neutral_rate_t):
                
                if 0 < self.inflation_rate[t-1] - self.inflation_rate[t-2] < 2:
                    neutral_rate_t += random.uniform(0.1, 0.3)  # 소폭 인상
                elif 2 <= self.inflation_rate[t-1] - self.inflation_rate[t-2] < 3:
                    neutral_rate_t += random.uniform(0.3, 0.5)  # 일반적인 인상
                elif self.inflation_rate[t-1] - self.inflation_rate[t-2] >= 3:
                    neutral_rate_t += random.uniform(0.5, 1.0)  # 대폭 인상

                reward += 2

            if (self.unemployment_change[t-3] < self.unemployment_change[t-2] < self.unemployment_change[t-1] and
                self.unit_labor_cost_change[t-3] > self.unit_labor_cost_change[t-2] > self.unit_labor_cost_change[t-1] and
                self.inflation_rate[t-3] > self.inflation_rate[t-2] > self.inflation_rate[t-1] and
                self.neutral_rate[t-1] > neutral_rate_t):

                if 0 < self.inflation_rate[t-2] - self.inflation_rate[t-1] < 2:
                    neutral_rate_t -= random.uniform(0.1, 0.3)  # 소폭 인하
                elif 2 <= self.inflation_rate[t-2] - self.inflation_rate[t-1] < 3:
                    neutral_rate_t -= random.uniform(0.3, 0.5)  # 일반적인 인하
                elif self.inflation_rate[t-2] - self.inflation_rate[t-1] >= 3:
                    neutral_rate_t -= random.uniform(0.5, 1.0)  # 대폭 인하

                reward += 2

            if (self.exchange_rate[t-3] < self.exchange_rate[t-2] < self.exchange_rate[t-1] and
                self.unit_labor_cost_change[t-3] < self.unit_labor_cost_change[t-2] < self.unit_labor_cost_change[t-1] and
                self.inflation_rate[t-3] < self.inflation_rate[t-2] < self.inflation_rate[t-1] and
                self.neutral_rate[t-1] < neutral_rate_t):

                if 0 < self.inflation_rate[t-1] - self.inflation_rate[t-2] < 2:
                    neutral_rate_t += random.uniform(0.1, 0.3)  # 소폭 인상
                elif 2 <= self.inflation_rate[t-1] - self.inflation_rate[t-2] < 3:
                    neutral_rate_t += random.uniform(0.3, 0.5)  # 일반적인 인상
                elif self.inflation_rate[t-1] - self.inflation_rate[t-2] >= 3:
                    neutral_rate_t += random.uniform(0.5, 1.0)  # 대폭 인상

                reward += 2

            if (self.exchange_rate[t-3] > self.exchange_rate[t-2] > self.exchange_rate[t-1] and
                self.unit_labor_cost_change[t-3] > self.unit_labor_cost_change[t-2] > self.unit_labor_cost_change[t-1] and
                self.inflation_rate[t-3] > self.inflation_rate[t-2] > self.inflation_rate[t-1] and
                self.neutral_rate[t-1] > neutral_rate_t):

                if 0 < self.inflation_rate[t-2] - self.inflation_rate[t-1] < 2:
                    neutral_rate_t -= random.uniform(0.1, 0.3)  # 소폭 인하
                elif 2 <= self.inflation_rate[t-2] - self.inflation_rate[t-1] < 3:
                    neutral_rate_t -= random.uniform(0.3, 0.5)  # 일반적인 인하
                elif self.inflation_rate[t-2] - self.inflation_rate[t-1] >= 3:
                    neutral_rate_t -= random.uniform(0.5, 1.0)  # 대폭 인하

                reward += 2    

            if (self.exchange_rate[t-3] < self.exchange_rate[t-2] < self.exchange_rate[t-1] and
                self.unit_labor_cost_change[t-3] > self.unit_labor_cost_change[t-2] > self.unit_labor_cost_change[t-1] and
                self.inflation_rate[t-3] < self.inflation_rate[t-2] < self.inflation_rate[t-1] and
                self.neutral_rate[t-1] > neutral_rate_t):

                if 0 < self.inflation_rate[t-1] - self.inflation_rate[t-2] < 2:
                    neutral_rate_t -= random.uniform(0.1, 0.3)  # 소폭 인하
                elif 2 <= self.inflation_rate[t-1] - self.inflation_rate[t-2] < 3:
                    neutral_rate_t -= random.uniform(0.3, 0.5)  # 일반적인 인하
                elif self.inflation_rate[t-1] - self.inflation_rate[t-2] >= 3:
                    neutral_rate_t -= random.uniform(0.5, 1.0)  # 대폭 인하

                reward += 2

            if (self.exchange_rate[t-3] > self.exchange_rate[t-2] > self.exchange_rate[t-1] and
                self.unit_labor_cost_change[t-3] < self.unit_labor_cost_change[t-2] < self.unit_labor_cost_change[t-1] and
                self.inflation_rate[t-3] > self.inflation_rate[t-2] > self.inflation_rate[t-1] and
                self.neutral_rate[t-1] < neutral_rate_t):

                if 0 < self.inflation_rate[t-2] - self.inflation_rate[t-1] < 2:
                    neutral_rate_t += random.uniform(0.1, 0.3)  # 소폭 인상
                elif 2 <= self.inflation_rate[t-2] - self.inflation_rate[t-1] < 3:
                    neutral_rate_t += random.uniform(0.3, 0.5)  # 일반적인 인상
                elif self.inflation_rate[t-2] - self.inflation_rate[t-1] >= 3:
                    neutral_rate_t += random.uniform(0.5, 1.0)  # 대폭 인상

                reward += 2

        self.current_step += 1
        if self.current_step < self.max_steps:
            next_state = [self.inflation_rate[self.current_step], self.interest_rate[self.current_step],
                          self.exchange_rate[self.current_step], self.unit_labor_cost_change[self.current_step],
                          self.unemployment_change[self.current_step]]
            done = False
        else:
            next_state = [0, 0, 0, 0, 0]  # 종료 상태
            done = True

        return np.array(next_state), reward, done



class DoubleDQNAgent:
    def __init__(self, input_dim, output_dim, total_episodes, buffer_capacity=50000, batch_size=128):
        self.model = DQN(input_dim, output_dim)
        self.target_model = DQN(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00005)
        self.criterion = nn.SmoothL1Loss()
        self.gamma = 0.95
        self.epsilon = 1.0  # 초기 epsilon 값을 1.0으로 낮춤
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999  # epsilon 감소 속도를 조금 더 완만하게 조정
        self.total_episodes = total_episodes
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(3)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def train(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        actions = np.array(actions)
        dones = np.array(dones)

        states_tensor = torch.FloatTensor(states)
        next_states_tensor = torch.FloatTensor(next_states)
        rewards_tensor = torch.FloatTensor(rewards)
        actions_tensor = torch.LongTensor(actions)
        dones_tensor = torch.FloatTensor(dones)

        q_values = self.model(states_tensor)
        next_q_values = self.model(next_states_tensor)
        next_q_target_values = self.target_model(next_states_tensor)

        q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        next_action = torch.argmax(next_q_values, 1)
        max_next_q_value = next_q_target_values.gather(1, next_action.unsqueeze(1)).squeeze(1)

        target_q_value = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q_value

        loss = self.criterion(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


# 이동 평균 함수 정의
def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# 강화 학습 실행
env = EconomicEnvironment(inflation_rate, interest_rate, exchange_rate, unit_labor_cost_change, unemployment_change)
total_episodes = 37
agent = DoubleDQNAgent(input_dim=5, output_dim=3, total_episodes=total_episodes)

total_rewards = []
scheduler = torch.optim.lr_scheduler.StepLR(agent.optimizer, step_size=30, gamma=0.95)  # 타겟 네트워크 업데이트 주기 증가

for e in range(total_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.replay_buffer.add((state, action, reward, next_state, done))
        agent.train()
        state = next_state
        episode_reward += reward
    total_rewards.append(episode_reward)
    if e % 25 == 0:  # 타겟 네트워크 업데이트 주기 증가
        agent.update_target_model()
    scheduler.step()

# 마지막 에피소드에서 받은 보상 출력
print(f"Last Episode ({total_episodes}): Total Reward = {total_rewards[-1]}")

# 에피소드 단계에 따른 획득 보상 변화
plt.figure(figsize=(10, 5))
plt.plot(range(total_episodes), total_rewards, color='black')
plt.xlabel('에피소드 수')
plt.ylabel('보상 수')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()

# 스무딩된 중립금리 값 계산
smoothed_neutral_rate = moving_average(env.neutral_rate, window_size=5)

# 결과 출력
print("\nYearly Estimated Neutral Interest Rate (Smoothed):")
smoothed_years = years[:len(smoothed_neutral_rate)]
for i in range(len(smoothed_years)):
    print(f"Year: {smoothed_years[i]}, Smoothed Neutral Interest Rate: {smoothed_neutral_rate[i]:.2f}%")

# 스무딩된 중립금리 그래프 (스무딩된 값만 표시)
plt.figure(figsize=(10, 5))
plt.plot(smoothed_years, smoothed_neutral_rate, label='Smoothed Neutral Rate', color='red')
plt.xlabel('Year')
plt.ylabel('Neutral Rate (%)')
plt.title('Smoothed Neutral Interest Rate over Time')
plt.grid(True)
plt.legend()
plt.xticks(np.arange(0, len(smoothed_years), step=10), rotation=45)
plt.tight_layout()
plt.show()



# 결과 값

# Last Episode (37): Total Reward = 74

# Yearly Estimated Neutral Interest Rate (Smoothed):
# Year: 1990/1Q, Smoothed Neutral Interest Rate: 4.85%
# Year: 1990/2Q, Smoothed Neutral Interest Rate: 5.15%
# Year: 1990/3Q, Smoothed Neutral Interest Rate: 5.01%
# Year: 1990/4Q, Smoothed Neutral Interest Rate: 4.76%
# Year: 1991/1Q, Smoothed Neutral Interest Rate: 4.82%
# Year: 1991/2Q, Smoothed Neutral Interest Rate: 4.82%
# Year: 1991/3Q, Smoothed Neutral Interest Rate: 4.42%
# Year: 1991/4Q, Smoothed Neutral Interest Rate: 4.42%
# Year: 1992/1Q, Smoothed Neutral Interest Rate: 3.66%
# Year: 1992/2Q, Smoothed Neutral Interest Rate: 3.20%
# Year: 1992/3Q, Smoothed Neutral Interest Rate: 2.85%
# Year: 1992/4Q, Smoothed Neutral Interest Rate: 2.45%
# Year: 1993/1Q, Smoothed Neutral Interest Rate: 2.25%
# Year: 1993/2Q, Smoothed Neutral Interest Rate: 2.31%
# Year: 1993/3Q, Smoothed Neutral Interest Rate: 2.31%
# Year: 1993/4Q, Smoothed Neutral Interest Rate: 2.34%
# Year: 1994/1Q, Smoothed Neutral Interest Rate: 2.34%
# Year: 1994/2Q, Smoothed Neutral Interest Rate: 2.14%
# Year: 1994/3Q, Smoothed Neutral Interest Rate: 2.13%
# Year: 1994/4Q, Smoothed Neutral Interest Rate: 2.13%
# Year: 1995/1Q, Smoothed Neutral Interest Rate: 2.29%
# Year: 1995/2Q, Smoothed Neutral Interest Rate: 2.32%
# Year: 1995/3Q, Smoothed Neutral Interest Rate: 2.52%
# Year: 1995/4Q, Smoothed Neutral Interest Rate: 2.76%
# Year: 1996/1Q, Smoothed Neutral Interest Rate: 2.84%
# Year: 1996/2Q, Smoothed Neutral Interest Rate: 2.60%
# Year: 1996/3Q, Smoothed Neutral Interest Rate: 2.57%
# Year: 1996/4Q, Smoothed Neutral Interest Rate: 2.37%
# Year: 1997/1Q, Smoothed Neutral Interest Rate: 2.13%
# Year: 1997/2Q, Smoothed Neutral Interest Rate: 2.05%
# Year: 1997/3Q, Smoothed Neutral Interest Rate: 1.65%
# Year: 1997/4Q, Smoothed Neutral Interest Rate: 1.25%
# Year: 1998/1Q, Smoothed Neutral Interest Rate: 0.85%
# Year: 1998/2Q, Smoothed Neutral Interest Rate: 0.40%
# Year: 1998/3Q, Smoothed Neutral Interest Rate: 0.00%
# Year: 1998/4Q, Smoothed Neutral Interest Rate: 0.32%
# Year: 1999/1Q, Smoothed Neutral Interest Rate: 0.91%
# Year: 1999/2Q, Smoothed Neutral Interest Rate: 1.37%
# Year: 1999/3Q, Smoothed Neutral Interest Rate: 1.77%
# Year: 1999/4Q, Smoothed Neutral Interest Rate: 2.56%
# Year: 2000/1Q, Smoothed Neutral Interest Rate: 2.63%
# Year: 2000/2Q, Smoothed Neutral Interest Rate: 2.92%
# Year: 2000/3Q, Smoothed Neutral Interest Rate: 2.84%
# Year: 2000/4Q, Smoothed Neutral Interest Rate: 2.64%
# Year: 2001/1Q, Smoothed Neutral Interest Rate: 2.06%
# Year: 2001/2Q, Smoothed Neutral Interest Rate: 1.91%
# Year: 2001/3Q, Smoothed Neutral Interest Rate: 1.34%
# Year: 2001/4Q, Smoothed Neutral Interest Rate: 1.21%
# Year: 2002/1Q, Smoothed Neutral Interest Rate: 1.42%
# Year: 2002/2Q, Smoothed Neutral Interest Rate: 1.50%
# Year: 2002/3Q, Smoothed Neutral Interest Rate: 1.40%
# Year: 2002/4Q, Smoothed Neutral Interest Rate: 1.24%
# Year: 2003/1Q, Smoothed Neutral Interest Rate: 1.26%
# Year: 2003/2Q, Smoothed Neutral Interest Rate: 1.01%
# Year: 2003/3Q, Smoothed Neutral Interest Rate: 0.83%
# Year: 2003/4Q, Smoothed Neutral Interest Rate: 0.87%
# Year: 2004/1Q, Smoothed Neutral Interest Rate: 0.77%
# Year: 2004/2Q, Smoothed Neutral Interest Rate: 0.55%
# Year: 2004/3Q, Smoothed Neutral Interest Rate: 0.45%
# Year: 2004/4Q, Smoothed Neutral Interest Rate: 0.49%
# Year: 2005/1Q, Smoothed Neutral Interest Rate: 0.56%
# Year: 2005/2Q, Smoothed Neutral Interest Rate: 0.76%
# Year: 2005/3Q, Smoothed Neutral Interest Rate: 1.12%
# Year: 2005/4Q, Smoothed Neutral Interest Rate: 1.42%
# Year: 2006/1Q, Smoothed Neutral Interest Rate: 1.57%
# Year: 2006/2Q, Smoothed Neutral Interest Rate: 1.61%
# Year: 2006/3Q, Smoothed Neutral Interest Rate: 1.82%
# Year: 2006/4Q, Smoothed Neutral Interest Rate: 1.81%
# Year: 2007/1Q, Smoothed Neutral Interest Rate: 2.07%
# Year: 2007/2Q, Smoothed Neutral Interest Rate: 2.17%
# Year: 2007/3Q, Smoothed Neutral Interest Rate: 2.58%
# Year: 2007/4Q, Smoothed Neutral Interest Rate: 2.22%
# Year: 2008/1Q, Smoothed Neutral Interest Rate: 1.62%
# Year: 2008/2Q, Smoothed Neutral Interest Rate: 0.82%
# Year: 2008/3Q, Smoothed Neutral Interest Rate: 0.22%
# Year: 2008/4Q, Smoothed Neutral Interest Rate: -0.70%
# Year: 2009/1Q, Smoothed Neutral Interest Rate: -0.75%
# Year: 2009/2Q, Smoothed Neutral Interest Rate: -0.42%
# Year: 2009/3Q, Smoothed Neutral Interest Rate: -0.37%
# Year: 2009/4Q, Smoothed Neutral Interest Rate: 0.02%
# Year: 2010/1Q, Smoothed Neutral Interest Rate: 0.26%
# Year: 2010/2Q, Smoothed Neutral Interest Rate: 0.58%
# Year: 2010/3Q, Smoothed Neutral Interest Rate: 0.73%
# Year: 2010/4Q, Smoothed Neutral Interest Rate: 1.10%
# Year: 2011/1Q, Smoothed Neutral Interest Rate: 0.96%
# Year: 2011/2Q, Smoothed Neutral Interest Rate: 0.97%
# Year: 2011/3Q, Smoothed Neutral Interest Rate: 0.60%
# Year: 2011/4Q, Smoothed Neutral Interest Rate: 0.28%
# Year: 2012/1Q, Smoothed Neutral Interest Rate: 0.08%
# Year: 2012/2Q, Smoothed Neutral Interest Rate: -0.07%
# Year: 2012/3Q, Smoothed Neutral Interest Rate: -0.22%
# Year: 2012/4Q, Smoothed Neutral Interest Rate: -0.06%
# Year: 2013/1Q, Smoothed Neutral Interest Rate: -0.11%
# Year: 2013/2Q, Smoothed Neutral Interest Rate: -0.22%
# Year: 2013/3Q, Smoothed Neutral Interest Rate: -0.06%
# Year: 2013/4Q, Smoothed Neutral Interest Rate: -0.16%
# Year: 2014/1Q, Smoothed Neutral Interest Rate: -0.57%
# Year: 2014/2Q, Smoothed Neutral Interest Rate: -0.77%
# Year: 2014/3Q, Smoothed Neutral Interest Rate: -0.99%
# Year: 2014/4Q, Smoothed Neutral Interest Rate: -1.11%
# Year: 2015/1Q, Smoothed Neutral Interest Rate: -0.85%
# Year: 2015/2Q, Smoothed Neutral Interest Rate: -0.95%
# Year: 2015/3Q, Smoothed Neutral Interest Rate: -1.00%
# Year: 2015/4Q, Smoothed Neutral Interest Rate: -1.05%
# Year: 2016/1Q, Smoothed Neutral Interest Rate: -0.71%
# Year: 2016/2Q, Smoothed Neutral Interest Rate: -0.84%
# Year: 2016/3Q, Smoothed Neutral Interest Rate: -0.84%
# Year: 2016/4Q, Smoothed Neutral Interest Rate: -0.46%
# Year: 2017/1Q, Smoothed Neutral Interest Rate: -0.41%
# Year: 2017/2Q, Smoothed Neutral Interest Rate: -0.99%
# Year: 2017/3Q, Smoothed Neutral Interest Rate: -0.87%
# Year: 2017/4Q, Smoothed Neutral Interest Rate: -0.77%
# Year: 2018/1Q, Smoothed Neutral Interest Rate: -0.46%
# Year: 2018/2Q, Smoothed Neutral Interest Rate: -0.41%
# Year: 2018/3Q, Smoothed Neutral Interest Rate: 0.34%
# Year: 2018/4Q, Smoothed Neutral Interest Rate: 0.14%
# Year: 2019/1Q, Smoothed Neutral Interest Rate: 0.86%
# Year: 2019/2Q, Smoothed Neutral Interest Rate: 0.88%
# Year: 2019/3Q, Smoothed Neutral Interest Rate: 0.63%
# Year: 2019/4Q, Smoothed Neutral Interest Rate: -0.12%
# Year: 2020/1Q, Smoothed Neutral Interest Rate: -0.46%
# Year: 2020/2Q, Smoothed Neutral Interest Rate: -0.76%
# Year: 2020/3Q, Smoothed Neutral Interest Rate: -1.07%
# Year: 2020/4Q, Smoothed Neutral Interest Rate: -0.53%
# Year: 2021/1Q, Smoothed Neutral Interest Rate: -0.00%
# Year: 2021/2Q, Smoothed Neutral Interest Rate: 0.25%
# Year: 2021/3Q, Smoothed Neutral Interest Rate: -0.02%
# Year: 2021/4Q, Smoothed Neutral Interest Rate: 0.18%
