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

# 데이터 입력
actual_gdp = np.array([109314800, 
                       111753200, 115366096, 117711800, 122126200, 124155296, 126818800, 129993800, 132308200, 133236600, 
                       133326704, 135407504, 138103008, 141645792, 144257296, 147017904, 151306208, 153774896, 156361696, 
                       162507392, 165904608, 169258896, 172772496, 176004304, 179362704, 182986096, 186204704, 189354496, 
                       190240800, 196995392, 198555104, 197649696, 184173904, 182685904, 185877696, 190517296, 196298800, 
                       204902896, 210601600, 216680096, 220786304, 223728304, 229891104, 229145200, 231953504, 234966304, 
                       238138800, 242336192, 248502400, 253005600, 258120896, 260953408, 259233296, 258833696, 263836096, 
                       270800000, 274198016, 276291488, 277412096, 279514688, 281893088, 287212608, 291536096, 294487808, 
                       299314816, 301501600, 306367712, 308755392, 313924096, 319355904, 323104608, 330073792, 331401984, 
                       333136800, 335854784, 324825696, 325118016, 329451104, 339310496, 341844704, 348541696, 355163296, 
                       359266208, 363646784, 367040096, 368841504, 370782784, 372534016, 375758912, 377885600, 379592800, 
                       381499296, 384777984, 389265088, 392597792, 396032704, 399424512, 402905888, 404192992, 406194112, 
                       409287808, 411277504, 417287488, 420167616, 421313088, 426505792, 428246592, 430814784, 435027104, 
                       438331200, 444445600, 443007616, 448561504, 451648704, 454433888, 457361216, 457051392, 462044608, 
                       463963584, 469606784, 463613408, 449633312, 460042112, 466234496, 474566112, 478745504, 479354592, 
                       486043712, 489254304, 492926016, 494078112, 492581184, 494206112, 497214112, 500277312, 503403296])

inflation = np.array([3.093, 2.963, 1.470, 1.087, 
                      3.911, 2.132, 1.906, 1.175, 1.758, 1.992, 0.976, -0.058, 1.650, 2.044, 0.740, 0.923, 
                      2.672, 1.390, 1.763, -0.127, 1.661, 1.456, 0.867, 0.397, 1.880, 1.651, 1.051, 0.409, 
                      1.506, 1.018, 0.980, 1.455, 5.258, 0.311, -0.085, 0.453, 0.029, 0.197, -0.028, 1.096, 
                      0.649, -0.134, 1.345, 0.662, 1.782, 1.165, 0.576, -0.254, 1.020, 1.326, 0.406, 0.527, 
                      1.762, 0.577, 0.272, 0.872, 1.491, 0.736, 1.166, -0.058, 1.374, 0.469, 0.536, 0.066, 
                      0.997, 0.624, 0.816, -0.324, 0.877, 1.064, 0.670, 0.728, 1.289, 2.080, 1.339, -0.271, 
                      0.723, 0.954, 0.564, 0.147, 1.290, 0.640, 0.770, 0.498, 1.883, 0.778, 1.094, 0.159, 
                      0.950, 0.208, 0.286, 0.263, 0.793, -0.130, 0.435, -0.031, 0.852, 0.344, 0.211, -0.424, 
                      0.568, 0.210, 0.293, -0.152, 0.507, 0.192, 0.188, 0.557, 1.248, -0.081, 0.480, -0.207, 
                      0.884, 0.330, 0.534, 0.029, -0.350, 0.440, -0.072, 0.279, 0.314, -0.510, 0.643, 0.010, 
                      1.287, 0.532, 0.686, 0.993, 1.648, 1.944, 1.135, 0.395, 1.053, 0.644, 1.009, 0.669])

interest_rate = np.array([7.000, 7.000, 7.000, 7.000, 
                          7.000, 7.000, 7.000, 7.000, 7.000, 7.000, 7.000, 7.000, 5.000, 5.000, 5.000, 5.000, 
                          5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 
                          5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 3.000, 3.000, 3.000, 3.000, 3.000, 3.000, 
                          5.000, 5.000, 5.000, 5.250, 5.000, 5.000, 4.000, 4.000, 4.000, 4.250, 4.250, 4.250, 
                          4.250, 4.000, 3.750, 3.750, 3.750, 3.750, 3.500, 3.250, 3.250, 3.250, 3.250, 3.750, 
                          4.000, 4.250, 4.500, 4.500, 4.500, 4.500, 5.000, 5.000, 5.000, 5.000, 5.250, 3.000, 
                          2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.250, 2.500, 3.000, 3.250, 3.250, 3.250, 
                          3.250, 3.250, 3.000, 2.750, 2.750, 2.500, 2.500, 2.500, 2.500, 2.500, 2.250, 2.000, 
                          1.750, 1.500, 1.500, 1.500, 1.500, 1.250, 1.250, 1.250, 1.250, 1.250, 1.250, 1.500, 
                          1.500, 1.500, 1.500, 1.750, 1.750, 1.750, 1.500, 1.250, 0.750, 0.500, 0.500, 0.500, 
                          0.500, 0.500, 0.750, 1.000, 1.250, 1.750, 2.500, 3.250, 3.500, 3.500, 3.500, 3.500])

exchange_rate = np.array([690.560, 710.260, 
                          715.480, 715.010, 721.570, 725.330, 733.010, 753.040, 766.500, 783.310, 787.390, 785.410, 
                          793.940, 799.520, 807.840, 808.890, 809.060, 807.260, 803.200, 795.560, 786.310, 763.290, 
                          765.710, 769.300, 783.000, 786.100, 816.860, 831.510, 865.320, 891.720, 898.240, 1140.920, 
                          1606.120, 1394.520, 1325.210, 1279.860, 1198.680, 1191.410, 1194.160, 1174.460, 1125.410, 
                          1116.290, 1115.270, 1164.110, 1271.680, 1305.680, 1293.780, 1291.840, 1319.630, 1270.680, 
                          1196.340, 1220.960, 1201.110, 1209.810, 1175.750, 1181.250, 1171.850, 1161.920, 1155.210, 
                          1093.970, 1022.480, 1007.960, 1029.190, 1037.000, 977.520, 950.410, 955.020, 938.400, 
                          938.900, 929.260, 928.170, 920.590, 955.970, 1016.720, 1062.640, 1362.790, 1415.220, 
                          1288.680, 1240.890, 1168.610, 1144.080, 1163.460, 1185.590, 1132.770, 1120.400, 1083.890, 
                          1083.040, 1144.750, 1131.470, 1151.810, 1133.540, 1090.860, 1084.080, 1122.150, 1112.180, 
                          1062.100, 1069.010, 1030.380, 1025.760, 1086.720, 1100.260, 1097.770, 1167.800, 1157.690, 
                          1201.440, 1163.300, 1121.370, 1157.370, 1154.280, 1129.430, 1132.240, 1105.720, 1072.290, 
                          1078.570, 1121.590, 1127.520, 1125.080, 1165.910, 1193.240, 1175.810, 1193.600, 1220.810, 
                          1188.540, 1117.640, 1114.050, 1121.230, 1157.350, 1183.170, 1204.950, 1259.570, 1337.980, 
                          1359.260, 1275.580, 1314.680, 1310.950, 1320.840])


industrial_production_index = np.array([10, 7, 15.8, 12.3, 
                                        6.4, 5.6, 2.7, 8, 10, 12.2, 2.3, 1.9, 3.5, 4.6, 10.5, 12.1, 10.8, 11.4, 8.2, 
                                        12.1, 13.8, 10.5, 11.7, 7.7, 6, 5.5, 6.1, 8, 6.1, 9.3, 8.8, -0.4, -7.9, -13.4, 
                                        -1.1, 6.6, 20.7, 32.4, 20.5, 24.8, 22.9, 16.9, 13.4, 3.2, 1.4, 1, 4.2, 1.5, 
                                        7, 2.2, 1.8, 11.2, 5.3, 9.1, 7.3, 11.2, 12.4, 13.1, 9.6, 4.2, 5.5, 4.3, 
                                        6.2, 10.7, 8.6, 9.3, 18.1, 2.2, 3.5, 6.7, -3, 9.4, 10.8, 7.3, 6.6, -18.4, 
                                        -10.1, -0.6, 12.1, 34.8, 22.8, 16, 1.7, 11.2, 8.8, 6, 7.1, 2.3, 0.9, 0.7, 
                                        0.3, 0.1, -1.8, -1.3, -3.4, 2.8, 3.1, 0.7, 2.1, 1.4, -1.2, 1, 4.6, -1.2, 
                                        2.3, 3.2, -1.2, 6, 5, 0.8, 10.8, -3.6, -2.3, 2.1, -6.5, 0.7, -2, -1.9, 2.1, 
                                        6.5, 6, -0.6, 8.1, 3.4, 6, 12.2, 0.1, 9.3, 5.4, 2.8, -1.6, -10.7, -5.9, 
                                        -4.5, 4.8, 6.3])

unemployment_rate = np.array([3.133333, 2.133333, 2.233333, 2.333333, 3.1, 2.2, 2.2, 2.3, 3.033333, 2.3, 2.333333, 
                              2.433333, 3.333333, 2.933333, 2.666667, 2.666667, 3.1, 2.466667, 2.233333, 2.1, 2.466667, 
                              2, 1.933333, 1.866667, 2.3, 2, 1.833333, 2.1, 3.1, 2.533333, 2.2, 2.633333, 5.8, 6.933333, 
                              7.533333, 7.533333, 8.5, 6.8, 6.033333, 5, 5.5, 4.166667, 4.033333, 4, 5.233333, 3.8, 3.5, 
                              3.466667, 4, 3.133333, 2.966667, 2.933333, 3.733333, 3.433333, 3.466667, 3.566667, 4.033333, 
                              3.533333, 3.5, 3.566667, 4.233333, 3.666667, 3.633333, 3.466667, 3.933333, 3.366667, 
                              3.333333, 3.266667, 3.6, 3.266667, 3.133333, 3.033333, 3.4, 3.133333, 3.066667, 3.1, 
                              3.833333, 3.8, 3.566667, 3.333333, 4.633333, 3.466667, 3.466667, 3.266667, 4.2, 3.4, 
                              3.1, 2.933333, 3.8, 3.266667, 3, 2.833333, 3.6, 3.1, 2.933333, 2.766667, 3.933333, 3.6, 
                              3.266667, 3.166667, 4.066667, 3.8, 3.4, 3.1, 4.266667, 3.7, 3.533333, 3.2, 4.233333, 
                              3.866667, 3.433333, 3.2, 4.266667, 3.933333, 3.766667, 3.366667, 4.5, 4.133333, 3.333333, 
                              3.166667, 4.133333, 4.333333, 3.566667, 3.733333, 4.966667, 3.933333, 2.833333, 2.966667, 
                              3.5, 3, 2.466667, 2.566667, 3.2, 2.733333, 2.333333, 2.566667])





# 년도와 분기
years = []
start_year = 1990
quarters = ['1Q', '2Q', '3Q', '4Q']
for i in range(len(actual_gdp)):
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
    def __init__(self, actual_gdp, inflation, interest_rate, exchange_rate, industrial_production_index):
        self.actual_gdp = actual_gdp
        self.inflation = inflation
        self.interest_rate = interest_rate
        self.exchange_rate = exchange_rate
        self.industrial_production_index = industrial_production_index
        self.unemployment_rate = unemployment_rate

        self.state = [108212600, 1.187, 7.000, 672.96, 2.9, 2.37]
        self.max_steps = len(actual_gdp)
        self.current_step = 0
        self.neutral_rate = np.zeros(len(actual_gdp))  # 중립금리 기록
        self.neutral_rate[0] = self.interest_rate[0]  # 중립금리 초기값 설정

    def reset(self):
        self.current_step = 0
        self.state = [self.actual_gdp[0], self.inflation[0], self.interest_rate[0], self.exchange_rate[0],
                      self.industrial_production_index[0], self.unemployment_rate[0]]
        return np.array(self.state)

    def step(self, action):
        t = self.current_step

        # 현재 시점의 경제 상태
        actual_gdp_t = self.actual_gdp[t]
        inflation_t = self.inflation[t]
        nominal_rate_t = self.interest_rate[t]
        exchange_rate_t = self.exchange_rate[t]
        industrial_production_index_t = self.industrial_production_index[t]
        unemployment_rate_t = self.unemployment_rate[t]
        neutral_rate_t = self.neutral_rate[t]

        # 행동에 따른 중립금리 조정
        if action == 0:  # 금리 상승
            neutral_rate_t += random.uniform(0.1, 0.5)
        elif action == 1:  # 금리 유지
            pass
        elif action == 2:  # 금리 하락
            neutral_rate_t -= random.uniform(0.1, 0.5)

        # 중립금리 제약 조건 적용 (명목금리의 ±3%p 범위 내)
        lower_bound = nominal_rate_t - 3.0
        upper_bound = nominal_rate_t + 3.0
        neutral_rate_t = min(max(neutral_rate_t, lower_bound), upper_bound)

        self.neutral_rate[t] = neutral_rate_t

        reward = 0
        # t=0,1일 때는 보상 조건 없이 패스
        if t == 0:
            self.current_step += 1
            return np.array([self.actual_gdp[self.current_step], self.inflation[self.current_step], self.interest_rate[self.current_step],
                             self.exchange_rate[self.current_step], self.industrial_production_index[self.current_step],
                             self.unemployment_rate[self.current_step]]), reward, False

        if t == 1:
            self.current_step += 1
            return np.array([self.actual_gdp[self.current_step], self.inflation[self.current_step], self.interest_rate[self.current_step],
                             self.exchange_rate[self.current_step], self.industrial_production_index[self.current_step],
                             self.unemployment_rate[self.current_step]]), reward, False

        if t >= 2:  # t-2 조건을 위한 체크
            # 1점 조건
            if (self.unemployment_rate[t-2] > self.unemployment_rate[t-1] and
                self.industrial_production_index[t-2] < self.industrial_production_index[t-1] and
                self.inflation[t-2] < self.inflation[t-1] and
                self.neutral_rate[t-1] < neutral_rate_t) or \
               (self.unemployment_rate[t-2] < self.unemployment_rate[t-1] and
                self.industrial_production_index[t-2] > self.industrial_production_index[t-1] and
                self.inflation[t-2] > self.inflation[t-1] and
                self.neutral_rate[t-1] > neutral_rate_t):
                reward += 1

            if (self.exchange_rate[t-2] < self.exchange_rate[t-1] and
                self.industrial_production_index[t-2] < self.industrial_production_index[t-1] and
                self.inflation[t-2] < self.inflation[t-1] and
                self.neutral_rate[t-1] < neutral_rate_t) or \
               (self.exchange_rate[t-2] > self.exchange_rate[t-1] and
                self.industrial_production_index[t-2] > self.industrial_production_index[t-1] and
                self.inflation[t-2] > self.inflation[t-1] and
                self.neutral_rate[t-1] > neutral_rate_t):
                reward += 1

            if (self.exchange_rate[t-2] < self.exchange_rate[t-1] and
                self.industrial_production_index[t-2] > self.industrial_production_index[t-1] and
                self.inflation[t-2] < self.inflation[t-1] and
                self.neutral_rate[t-1] > neutral_rate_t) or \
               (self.exchange_rate[t-2] > self.exchange_rate[t-1] and
                self.industrial_production_index[t-2] < self.industrial_production_index[t-1] and
                self.inflation[t-2] > self.inflation[t-1] and
                self.neutral_rate[t-1] < neutral_rate_t):
                reward += 1

        if t >= 3:  # t-3 조건을 위한 체크
            # 2점 조건
            if (self.unemployment_rate[t-3] > self.unemployment_rate[t-2] > self.unemployment_rate[t-1] and
                self.industrial_production_index[t-3] < self.industrial_production_index[t-2] < self.industrial_production_index[t-1] and
                self.inflation[t-3] < self.inflation[t-2] < self.inflation[t-1] and
                self.neutral_rate[t-1] < neutral_rate_t) or \
               (self.unemployment_rate[t-3] < self.unemployment_rate[t-2] < self.unemployment_rate[t-1] and
                self.industrial_production_index[t-3] > self.industrial_production_index[t-2] > self.industrial_production_index[t-1] and
                self.inflation[t-3] > self.inflation[t-2] > self.inflation[t-1] and
                self.neutral_rate[t-1] > neutral_rate_t):
                reward += 2

            if (self.exchange_rate[t-3] < self.exchange_rate[t-2] < self.exchange_rate[t-1] and
                self.industrial_production_index[t-3] < self.industrial_production_index[t-2] < self.industrial_production_index[t-1] and
                self.inflation[t-3] < self.inflation[t-2] < self.inflation[t-1] and
                self.neutral_rate[t-1] < neutral_rate_t) or \
               (self.exchange_rate[t-3] > self.exchange_rate[t-2] > self.exchange_rate[t-1] and
                self.industrial_production_index[t-3] > self.industrial_production_index[t-2] > self.industrial_production_index[t-1] and
                self.inflation[t-3] > self.inflation[t-2] > self.inflation[t-1] and
                self.neutral_rate[t-1] > neutral_rate_t):
                reward += 2

            if (self.exchange_rate[t-3] < self.exchange_rate[t-2] < self.exchange_rate[t-1] and
                self.industrial_production_index[t-3] > self.industrial_production_index[t-2] > self.industrial_production_index[t-1] and
                self.inflation[t-3] < self.inflation[t-2] < self.inflation[t-1] and
                self.neutral_rate[t-1] > neutral_rate_t) or \
               (self.exchange_rate[t-3] > self.exchange_rate[t-2] > self.exchange_rate[t-1] and
                self.industrial_production_index[t-3] < self.industrial_production_index[t-2] < self.industrial_production_index[t-1] and
                self.inflation[t-3] > self.inflation[t-2] > self.inflation[t-1] and
                self.neutral_rate[t-1] < neutral_rate_t):
                reward += 2

        self.current_step += 1
        if self.current_step < self.max_steps:
            next_state = [self.actual_gdp[self.current_step], self.inflation[self.current_step], self.interest_rate[self.current_step],
                          self.exchange_rate[self.current_step], self.industrial_production_index[self.current_step],
                          self.unemployment_rate[self.current_step]]
            done = False
        else:
            next_state = [0, 0, 0, 0, 0, 0]  # 종료 상태
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
env = EconomicEnvironment(actual_gdp, inflation, interest_rate, exchange_rate, industrial_production_index)
total_episodes = 45
agent = DoubleDQNAgent(input_dim=6, output_dim=3, total_episodes=total_episodes)

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
