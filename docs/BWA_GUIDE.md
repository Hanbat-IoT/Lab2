# BWA (Bandwidth Allocation) 알고리즘 가이드

## 📚 개요

BWA는 DRL(Deep Reinforcement Learning) 기반의 동적 배치 크기 최적화 알고리즘입니다. 각 라운드마다 최적의 배치 크기를 자동으로 선택하여 학습 효율을 향상시킵니다.

## 🎯 주요 특징

### 1. **동적 배치 크기 최적화**
- 고정된 배치 크기 대신 상황에 맞게 자동 조정
- 학습 상태(손실, 정확도, 시간)를 고려한 최적화

### 2. **DRL 기반 학습**
- Actor-Critic 아키텍처
- PPO (Proximal Policy Optimization) 알고리즘
- 경험 기반 학습으로 점진적 개선

### 3. **보상 함수**
```python
reward = (
    α × accuracy_improvement +
    γ × loss_improvement -
    β × time_cost
) × λ_k
```

## 🚀 사용 방법

### 1. 서버 실행

```bash
# BWA 전략으로 서버 시작
python flower_server.py \
    --strategy fedavg_bwa \
    --num_clients 3 \
    --num_rounds 20 \
    --dataset mnist
```

### 2. 클라이언트 실행

```bash
# 각 디바이스에서 클라이언트 실행
python flower_client.py \
    --client_id 0 \
    --server_address <SERVER_IP>:8080 \
    --dataset mnist
```

### 3. 결과 분석

```bash
# FedAvg vs FedAvg+BWA 비교
python scripts/compare_strategies.py \
    --baseline results_fedavg_mnist_3clients_20241202_120000.json \
    --adm results_fedavg_bwa_mnist_3clients_20241202_120500.json
```

## 📊 전략 비교

| 전략 | 배치 크기 | 최적화 방식 | 주요 장점 |
|------|----------|------------|----------|
| **FedAvg** | 고정 (32) | 없음 | 단순, 안정적 |
| **FedAvg+ADM** | 고정 (32) | 데이터 사용량 최적화 | 시간 효율 |
| **FedAvg+BWA** | 동적 (16~128) | 배치 크기 최적화 | 학습 효율 |

## 🔧 파라미터 설정

### BWA 알고리즘 파라미터

```python
BWAAlgorithm(
    num_clients=3,                      # 클라이언트 수
    batch_size_options=[16, 32, 64, 128],  # 선택 가능한 배치 크기
    learning_rate_actor=1e-4,           # Actor 학습률
    learning_rate_critic=1e-3,          # Critic 학습률
    gamma=0.99,                         # 할인 계수
    ppo_epochs=10,                      # PPO 업데이트 횟수
    ppo_clip=0.2                        # PPO 클리핑 파라미터
)
```

### 보상 함수 가중치

```python
alpha = 1.0   # 정확도 가중치
beta = 0.1    # 시간 비용 가중치
gamma = 0.5   # 손실 개선 가중치
```

## 📈 결과 파일

### JSON 결과 파일
```json
{
  "strategy": "FedAvg+BWA",
  "dataset": "mnist",
  "num_clients": 3,
  "num_rounds": 20,
  "accuracies": [0.75, 0.82, 0.87, ...],
  "round_times": [12.5, 11.3, 10.8, ...],
  "batch_size_history": [32, 64, 64, 128, 64, ...],
  "total_time": 225.6,
  "avg_round_time": 11.28
}
```

### BWA 모델 파일
- `results_fedavg_bwa_mnist_3clients_20241202_120500_bwa_actor.pth`
- `results_fedavg_bwa_mnist_3clients_20241202_120500_bwa_critic.pth`

## 📊 생성되는 그래프

### 1. 정확도 비교
`comparison_accuracy_mnist.png`
- FedAvg vs FedAvg+BWA 정확도 비교

### 2. 훈련 시간 비교
`comparison_training_time_mnist.png`
- 라운드별 시간 및 누적 시간 비교

### 3. 배치 크기 변화
`batch_size_evolution_mnist.png`
- BWA가 선택한 배치 크기의 변화 추이

## 🎓 알고리즘 동작 원리

### 1. 상태 (State)
```python
state = [
    loss,              # 현재 손실
    accuracy,          # 현재 정확도
    round_time,        # 라운드 시간
    data_dist_1,       # 클라이언트 1 데이터 분포
    data_dist_2,       # 클라이언트 2 데이터 분포
    ...
]
```

### 2. 행동 (Action)
```python
# 배치 크기 선택
action = select_batch_size([16, 32, 64, 128])
```

### 3. 보상 (Reward)
```python
# 성능 개선과 시간 비용의 균형
reward = performance_gain - time_cost
```

### 4. 학습 과정
```
1. 현재 상태 관찰
2. Actor 네트워크가 배치 크기 선택
3. 선택된 배치 크기로 FL 라운드 실행
4. 보상 계산 및 경험 저장
5. Critic 네트워크로 상태 가치 평가
6. PPO로 Actor 네트워크 업데이트
7. MSE로 Critic 네트워크 업데이트
```

## 💡 사용 팁

### 1. 배치 크기 옵션 설정
```python
# 작은 배치 크기: 빠른 업데이트, 불안정
# 큰 배치 크기: 안정적, 느린 업데이트
batch_size_options=[16, 32, 64, 128, 256]
```

### 2. 학습률 조정
```python
# Actor: 정책 변화 속도
learning_rate_actor=1e-4  # 너무 크면 불안정

# Critic: 가치 추정 속도
learning_rate_critic=1e-3  # Actor보다 크게 설정
```

### 3. 보상 함수 튜닝
```python
# 정확도 중시
alpha = 2.0, beta = 0.1, gamma = 0.5

# 시간 효율 중시
alpha = 1.0, beta = 0.5, gamma = 0.5

# 균형
alpha = 1.0, beta = 0.1, gamma = 0.5
```

## 🔍 디버깅

### 로그 확인
```bash
# BWA 선택 로그
[BWA] Selected batch size: 64
[BWA] Reward: 0.0234, Buffer size: 45
[BWA] Actor Loss: 0.0123, Critic Loss: 0.0456
```

### 모델 저장/로드
```python
# 저장
bwa.save_models("my_bwa_model")

# 로드
bwa.load_models("my_bwa_model")
```

## 📚 참고 자료

### 관련 알고리즘
- **PPO**: Proximal Policy Optimization
- **Actor-Critic**: 정책 기반 + 가치 기반 강화학습
- **Experience Replay**: 경험 재사용

### 관련 파일
- `BWA.py`: BWA 알고리즘 구현
- `flower_server.py`: Flower 서버 통합
- `scripts/compare_strategies.py`: 결과 비교 스크립트

## ⚠️ 주의사항

1. **초기 학습 단계**: 처음 몇 라운드는 랜덤하게 선택될 수 있음
2. **경험 버퍼**: 충분한 경험이 쌓여야 학습 시작 (최소 32개)
3. **하이퍼파라미터**: 데이터셋과 환경에 따라 조정 필요
4. **계산 비용**: DRL 학습으로 인한 추가 오버헤드 존재

## 🎯 예상 효과

### 장점
- ✅ 동적 배치 크기로 학습 효율 향상
- ✅ 상황에 맞는 최적화
- ✅ 점진적 성능 개선

### 단점
- ⚠️ 초기 학습 필요
- ⚠️ 추가 계산 비용
- ⚠️ 하이퍼파라미터 튜닝 필요

## 📞 문제 해결

### Q1: BWA가 항상 같은 배치 크기를 선택해요
**A**: 학습이 충분하지 않거나 보상 함수 조정이 필요합니다.

### Q2: 학습이 불안정해요
**A**: learning_rate_actor를 낮추거나 ppo_clip을 조정하세요.

### Q3: 성능이 FedAvg보다 낮아요
**A**: 초기 학습 단계일 수 있습니다. 더 많은 라운드를 실행하세요.

---

**작성일**: 2024-12-02  
**버전**: 1.0.0
