# 🚀 Federated Learning with ADM & BWA Optimization

이질적인 IoT 환경에서 **ADM (Adaptive Data Management)**과 **BWA (Bandwidth Allocation)** 알고리즘을 활용한 연합학습 최적화 프로젝트입니다.

## 📋 프로젝트 개요

### 🎯 핵심 기능
- **3가지 연합학습 전략**: FedAvg (Baseline), FedAvg+ADM, FedAvg+BWA
- **ADM (Adaptive Data Management)**: 클라이언트 성능에 따른 데이터 비율(v_n) 최적화
- **BWA (Bandwidth Allocation)**: PPO 기반 동적 배치 크기 최적화
- **실시간 Calibration**: 실제 학습 시간 기반 파라미터 자동 보정
- **IID/Non-IID 지원**: 다양한 데이터 분포 환경 실험 (bias 조절 가능)

## 🛠️ 기술 스택

### **Backend Framework**
- **Python 3.8+**: 메인 개발 언어
- **Flower 0.18.0**: 연합학습 프레임워크
- **PyTorch 2.0+**: 딥러닝 프레임워크

### **최적화 & 수학**
- **CVXPy**: Convex Optimization (ADM)
- **NumPy**: 수치 계산
- **PPO (Proximal Policy Optimization)**: DRL 기반 BWA

### **데이터 & 시각화**
- **TorchVision**: 데이터셋 (MNIST, CIFAR-10)
- **Matplotlib**: 결과 시각화
- **Pandas**: 데이터 분석

### **개발 도구**
- **Git**: 버전 관리
- **Logging**: 상세한 실험 로그
- **Argparse**: CLI 인터페이스

## 📁 디렉토리 구조

```
FL_GUI/
│
├── 📄 flower_server.py            # Flower 서버 (메인)
├── 📄 flower_client.py            # Flower 클라이언트 (메인)
│
├── 📁 src/                        # 소스 코드
│   ├── __init__.py
│   ├── 📁 algorithms/             # 알고리즘
│   │   ├── __init__.py
│   │   ├── ADM.py                # Adaptive Data Management
│   │   └── BWA.py                # Bandwidth Allocation (PPO)
│   ├── models.py                 # CNN 모델 정의
│   ├── utils.py                  # 데이터 로더 (IID/Non-IID)
│   ├── updateModel.py            # 학습/평가 함수
│   └── options.py                # 설정 파서
│
├── 📁 scripts/                    # 유틸리티 스크립트
│   ├── compare_strategies.py     # 그래프 생성 & 성능 비교
│   ├── check_versions.py         # 의존성 버전 확인
│   ├── deploy_files.sh           # 파일 배포
│   └── setup_environment.sh      # 환경 설정
│
├── 📁 configs/                    # 설정 파일
│   ├── requirements.txt          # Python 의존성
│   └── README_DOCKER.md          # Docker 가이드
│
├── 📁 docs/                       # 문서
│   ├── README.md                 # 프로젝트 설명
│   └── FLOWER_GUIDE.md           # Flower 가이드
│
├── 📁 data/                       # 데이터셋 (자동 다운로드)
│   └── MNIST/, CIFAR10/
│
├── 📁 logs/                       # 학습 로그
│   └── *.log
│
├── 📁 venv/                       # 가상환경 (gitignore)
├── 📁 __pycache__/                # 파이썬 캐시 (gitignore)
│
├── .git/                         # Git 저장소
├── .gitignore                    # Git 제외 파일
├── README.md                     # 이 파일
└── PROJECT_STRUCTURE.md          # 프로젝트 구조 상세
```

## 📦 디바이스별 필요 파일

### 서버 (메인 노드)
```
메인 파일:
  - flower_server.py ⭐

소스 코드:
  - src/algorithms/ADM.py
  - src/algorithms/BWA.py
  - src/models.py
  - src/utils.py
  - src/updateModel.py
  - src/options.py

유틸리티:
  - scripts/compare_strategies.py (그래프 생성)
  - scripts/check_versions.py

설정:
  - configs/requirements.txt
```

### 클라이언트 (각 노드)
```
메인 파일:
  - flower_client.py ⭐

소스 코드:
  - src/models.py
  - src/utils.py
  - src/updateModel.py

유틸리티:
  - scripts/check_versions.py

설정:
  - configs/requirements.txt
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 의존성 설치
pip install -r configs/requirements.txt

# 버전 확인
python scripts/check_versions.py
```

### 2. 서버 실행
```bash
# FedAvg Baseline
python flower_server.py --strategy fedavg --num_clients 3 --num_rounds 20

# FedAvg + ADM
python flower_server.py --strategy fedavg_adm --num_clients 3 --num_rounds 20

# FedAvg + BWA
python flower_server.py --strategy fedavg_bwa --num_clients 3 --num_rounds 20
```

### 3. 클라이언트 실행 (각 디바이스)
```bash
# IID 데이터 분포
python flower_client.py --client_id 0 --server_address <SERVER_IP>:8080 --iid

# Non-IID 50% 편향 (기본값)
python flower_client.py --client_id 0 --server_address <SERVER_IP>:8080

# Non-IID 100% 편향 (극단적)
python flower_client.py --client_id 0 --server_address <SERVER_IP>:8080 --bias 1.0
```

### 4. 결과 분석 & 그래프 생성
```bash
python scripts/compare_strategies.py \
    --baseline results_fedavg_mnist_3clients_20250114_120000.json \
    --proposed results_fedavg_adm_mnist_3clients_20250114_120500.json
```

## 🔧 유틸리티

### 버전 확인
```bash
python scripts/check_versions.py
```

### 파일 배포 (Linux/Mac)
```bash
chmod +x scripts/deploy_files.sh scripts/setup_environment.sh
./scripts/deploy_files.sh
```

### 환경 자동 설정 (Linux/Mac)
```bash
./scripts/setup_environment.sh
```

## 📝 로그 파일

학습 로그는 자동으로 `logs/` 디렉토리에 저장됩니다:
- 형식: `[rounds]rounds_[clients]clients_[dataset]_[IID].log`
- 예시: `20rounds_3clients_mnist_1IID.log`

## 🔍 주요 파일 설명

| 파일 | 설명 | 용도 |
|-----|------|-----|
| `flower_server.py` | Flower 서버 구현 | FL 서버 (3가지 전략 지원) |
| `flower_client.py` | Flower 클라이언트 구현 | FL 클라이언트 (IID/Non-IID) |
| `src/algorithms/ADM.py` | ADM 알고리즘 | 클라이언트별 v_n 최적화 |
| `src/algorithms/BWA.py` | BWA 알고리즘 | PPO 기반 배치 크기 최적화 |
| `src/models.py` | CNN 모델 | MNIST/CIFAR 모델 정의 |
| `src/utils.py` | 데이터 로더 | Loader/BiasLoader/ShardLoader |
| `src/updateModel.py` | 학습/평가 | train() / test() 함수 |
| `scripts/compare_strategies.py` | 그래프 생성 | 성능 비교 시각화 |

## 🔧 고급 설정

### Non-IID 데이터 분포

**데이터 분포 예시 (MNIST, 2500 샘플):**

| Bias | Client 0 선호 클래스 (0,1) | 나머지 클래스 (2~9) |
|------|--------------------------|-------------------|
| 0.5 (50%) | 625개 × 2 = 1250개 | 156개 × 8 = 1250개 |
| 0.7 (70%) | 875개 × 2 = 1750개 | 94개 × 8 = 750개 |
| 1.0 (100%) | 1250개 × 2 = 2500개 | 0개 |

```bash
# IID (균등 분포)
python flower_client.py --client_id 0 --iid

# Non-IID 50% 편향 (기본값)
python flower_client.py --client_id 0 --bias 0.5

# Non-IID 100% 편향 (극단적)
python flower_client.py --client_id 0 --bias 1.0
```

## 📚 참고 자료

- **Flower Documentation**: https://flower.dev/
- **ADM Paper**: [Adaptive Data Management for Federated Learning]
- **BWA Paper**: [Bandwidth-Aware Federated Learning with DRL]
- **PyTorch Federated Learning**: https://pytorch.org/tutorials/

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
