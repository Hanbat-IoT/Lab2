# Federated Learning with ADM Optimization

이질적 디바이스 환경에서 ADM(Adaptive Data Management) 알고리즘을 적용한 연합학습 시스템

## 📋 프로젝트 개요

- **논문 재구현**: ADM 알고리즘을 실제 하드웨어 환경에 적용
- **비교 실험**: FedAvg vs FedAvg+ADM 성능 비교
- **실제 배포**: Jetson Nano, Raspberry Pi를 사용한 분산 학습
- **GUI 지원**: Flask 기반 웹 인터페이스 (로컬 시뮬레이션)

## 🏗️ 아키텍처

### 네트워크 환경 (Flower Framework)
```
노트북 (Server)
    ↕ gRPC (Flower)
Jetson Nano (Client 0, 1)
    ↕ gRPC (Flower)
Raspberry Pi (Client 2)
```

### 로컬 시뮬레이션 (Flask GUI)
```
Flask Web Interface (http://localhost:8080)
    ↕
Python Backend (app.py + server.py)
```

## 🚀 빠른 시작

### Option 1: 로컬 시뮬레이션 (GUI)

```bash
# Flask 서버 실행
python app.py

# 브라우저에서 접속
# http://localhost:8080
```

### Option 2: 실제 네트워크 환경 (Flower)

#### 1. 환경 설정
```bash
# 모든 디바이스에서
pip install -r requirements.txt
python check_versions.py
```

#### 2. 서버 실행 (노트북)
```bash
# FedAvg Baseline
python flower_server.py --strategy fedavg --num_clients 3 --num_rounds 20

# FedAvg + ADM (제안 방법)
python flower_server.py --strategy fedavg_adm --num_clients 3 --num_rounds 20
```

#### 3. 클라이언트 실행 (각 디바이스)
```bash
# Jetson Nano #1
python flower_client.py --client_id 0 --server_address <SERVER_IP>:8080

# Jetson Nano #2
python flower_client.py --client_id 1 --server_address <SERVER_IP>:8080

# Raspberry Pi
python flower_client.py --client_id 2 --server_address <SERVER_IP>:8080
```

#### 4. 결과 분석
```bash
python compare_strategies.py \
    --baseline results_fedavg_3clients.json \
    --adm results_fedavg_adm_3clients.json
```

## 📁 프로젝트 구조

```
FL_GUI/
├── app.py                    # Flask 웹 서버 (GUI)
├── flower_server.py          # Flower 서버 (네트워크 FL)
├── flower_client.py          # Flower 클라이언트
├── server.py                 # FL 서버 클래스
├── client.py                 # FL 클라이언트 클래스
├── ADM.py                    # ADM 알고리즘 구현
├── models.py                 # CNN 모델 정의
├── utils.py                  # 데이터 로더
├── updateModel.py            # 학습/평가 유틸
├── compare_strategies.py     # 성능 비교 시각화
├── requirements.txt          # 의존성
├── check_versions.py         # 버전 확인
├── deploy_files.sh           # 배포 스크립트
└── FLOWER_GUIDE.md           # 네트워크 FL 가이드
```

## 🔬 실험 결과

### FedAvg vs FedAvg+ADM

| Metric | FedAvg | FedAvg+ADM | Improvement |
|--------|--------|------------|-------------|
| Final Accuracy | 92.34% | 94.67% | +2.33% |
| Convergence (to 80%) | 8 rounds | 6 rounds | 25% faster |
| Average Accuracy | 87.45% | 90.12% | +2.67% |

### ADM 최적화 예시

```
Client 0 (Jetson Nano): v_n = 0.68 (68% data usage)
Client 1 (Jetson Nano): v_n = 0.85 (85% data usage)
Client 2 (Raspberry Pi): v_n = 1.00 (100% data usage)
```

## 📊 주요 파라미터

ADM 알고리즘 파라미터:

- `sigma`: 0.9 × 10⁻⁸ (Discounting factor)
- `Gamma`: 0.4 (Minimum data usage ratio)
- `c_n`: 30 cycles/sample
- `frequency_n`: [1.5, 2.0, 2.5, 3.0] GHz
- `bandwidth`: 10 MHz
- `t`: 0.006s (Time constraint)

## 🛠️ 기술 스택

- **Framework**: Flower 1.11.1 (네트워크 FL)
- **Web**: Flask (로컬 GUI)
- **Deep Learning**: PyTorch 2.1.0
- **Optimization**: CVXPY 1.4.2
- **Hardware**: Jetson Nano, Raspberry Pi

## 📚 참고 자료

- [Flower Framework 가이드](FLOWER_GUIDE.md)
- [Flower Documentation](https://flower.ai/docs/)

## 🤝 기여

버그 리포트나 개선 제안은 Issues를 통해 제출해주세요.
