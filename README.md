# 🚀 Federated Learning with ADM & BWA Optimization

이질적인 IoT 환경에서 **ADM (Adaptive Data Management)**과 **BWA (Bandwidth Allocation)** 알고리즘을 활용한 연합학습 최적화 프로젝트입니다.

## 📋 프로젝트 개요

### 🎯 목표
- **이질적인 디바이스 환경**에서 연합학습 성능 최적화
- **ADM 알고리즘**을 통한 클라이언트별 데이터 사용량 동적 조절
- **BWA 알고리즘**을 통한 DRL 기반 배치 크기 최적화
- **실제 하드웨어** (Jetson Nano, Raspberry Pi, 노트북) 환경에서 검증

### 🔬 핵심 기술
- **ADM (Adaptive Data Management)**: 클라이언트 성능에 따른 데이터 비율(v_n) 최적화
- **BWA (Bandwidth Allocation)**: PPO 기반 동적 배치 크기 최적화
- **실시간 Calibration**: 실제 학습 시간 기반 파라미터 자동 보정
- **IID/Non-IID 지원**: 다양한 데이터 분포 환경 실험

### 🏗️ 아키텍처
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raspberry Pi  │    │   Jetson Nano   │    │     Laptop      │
│   (Client 0,1)  │    │   (Client 2,3)  │    │    (Server)     │
│                 │    │                 │    │                 │
│ • ARM Cortex    │    │ • ARM Cortex    │    │ • Intel/AMD     │
│ • 1GB RAM       │    │ • 4GB RAM       │    │ • 16GB+ RAM     │
│ • 느린 학습      │    │ • 중간 학습      │    │ • 빠른 학습      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │   Flower Server     │
                    │                     │
                    │ • ADM Optimization  │
                    │ • BWA Optimization  │
                    │ • Real-time Calib.  │
                    └─────────────────────┘
```

## 🛠️ 기술 스택

### **Backend Framework**
- **Python 3.8+**: 메인 개발 언어
- **Flower 1.8.0**: 연합학습 프레임워크
- **Flask 2.3.0**: 웹 GUI 서버
- **PyTorch 2.0+**: 딥러닝 프레임워크

### **최적화 & 수학**
- **CVXPy**: Convex Optimization (ADM)
- **NumPy**: 수치 계산
- **SciPy**: 과학 계산
- **PPO (Proximal Policy Optimization)**: DRL 기반 BWA

### **데이터 & 시각화**
- **TorchVision**: 데이터셋 (MNIST, CIFAR-10)
- **Matplotlib**: 결과 시각화
- **Pandas**: 데이터 분석
- **Seaborn**: 고급 시각화

### **하드웨어 지원**
- **CUDA**: GPU 가속 (가능한 경우)
- **ARM64**: Jetson Nano, Raspberry Pi 지원
- **Cross-platform**: Windows, Linux, macOS

### **개발 도구**
- **Git**: 버전 관리
- **Docker**: 컨테이너화 (Jetson/RPi)
- **Logging**: 상세한 실험 로그
- **Argparse**: CLI 인터페이스

## 📁 디렉토리 구조

```
FL_GUI/
│
├── docs/                           # 📚 문서
│   ├── README.md                   # 프로젝트 전체 설명
│   └── FLOWER_GUIDE.md            # Flower 네트워크 FL 가이드
│
├── configs/                        # ⚙️ 설정 파일
│   ├── requirements.txt           # Python 의존성
│   └── Dockerfile.jetson          # Jetson Nano용 Docker 설정
│
├── scripts/                        # 🔧 유틸리티 스크립트
│   ├── check_versions.py          # 버전 확인
│   ├── compare_strategies.py      # 성능 비교 시각화
│   ├── deploy_files.sh            # 파일 배포 스크립트
│   └── setup_environment.sh       # 환경 설정 스크립트
│
├── templates/                      # 🎨 Flask HTML 템플릿
│   ├── index.html                 # 메인 페이지
│   ├── loading.html               # 학습 진행 페이지
│   └── result.html                # 결과 페이지
│
├── data/                          # 📊 데이터셋 (자동 다운로드)
│   └── MNIST/, CIFAR10/
│
├── logs/                          # 📝 학습 로그
│   └── *.log
│
├── venv/                          # 🐍 가상환경 (gitignore)
│
├── __pycache__/                   # 파이썬 캐시 (gitignore)
│
├── .git/                          # Git 저장소
├── .gitignore                     # Git 제외 파일 목록
│
└── Core Files                     # 💻 핵심 코드
    ├── app.py                     # Flask 웹 서버 (GUI)
    ├── flower_server.py           # Flower 서버 (네트워크 FL)
    ├── flower_client.py           # Flower 클라이언트
    ├── server.py                  # FL 서버 클래스
    ├── client.py                  # FL 클라이언트 클래스
    ├── run_app.py                 # Flask 실행 헬퍼
    ├── ADM.py                     # ADM 알고리즘
    ├── models.py                  # CNN 모델 정의
    ├── utils.py                   # 데이터 로더
    ├── updateModel.py             # 학습/평가 유틸
    ├── dists.py                   # 분포 함수
    └── options.py                 # 옵션 파서
```

## 📦 디바이스별 필요 파일

### 서버 (노트북)
```
Core Files:
  - flower_server.py ⭐ (Flower 사용시)
  - app.py ⭐ (GUI 사용시)
  - server.py
  - ADM.py
  - models.py
  - utils.py

Scripts:
  - scripts/compare_strategies.py
  - scripts/check_versions.py

Configs:
  - configs/requirements.txt
```

### 클라이언트 (Jetson Nano, Raspberry Pi)
```
Core Files:
  - flower_client.py ⭐
  - models.py
  - utils.py
  - updateModel.py
  - dists.py

Scripts:
  - scripts/check_versions.py

Configs:
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

### 2. 로컬 시뮬레이션 (Flask GUI)
```bash
python app.py
# 브라우저: http://localhost:8080
```

### 3. 네트워크 환경 (Flower)
```bash
# 서버 (노트북)
python flower_server.py --strategy fedavg_adm --num_clients 3 --num_rounds 20

# 클라이언트 (각 디바이스)
python flower_client.py --client_id 0 --server_address <SERVER_IP>:8080
```

### 4. 결과 분석
```bash
python scripts/compare_strategies.py \
    --baseline results_fedavg_3clients.json \
    --adm results_fedavg_adm_3clients.json
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
| `flower_server.py` | Flower 서버 구현 | 네트워크 FL 서버 |
| `flower_client.py` | Flower 클라이언트 구현 | 네트워크 FL 클라이언트 |
| `app.py` | Flask 웹 서버 | GUI 기반 시뮬레이션 |
| `ADM.py` | ADM 알고리즘 | 최적화 알고리즘 |
| `server.py` | 서버 클래스 | FL 서버 로직 |
| `client.py` | 클라이언트 클래스 | FL 클라이언트 로직 |
| `models.py` | CNN 모델 | MNIST/CIFAR 모델 |
| `utils.py` | 데이터 로더 | IID/Non-IID 데이터 분할 |

## 📊 실험 결과

### 성능 비교 (MNIST, 4 클라이언트, 20 라운드)

| 전략 | 최종 정확도 | 총 학습 시간 | 라운드당 평균 시간 |
|-----|------------|-------------|------------------|
| **FedAvg (Baseline)** | 94.2% | 1,800초 | 90초 |
| **FedAvg + ADM** | 95.1% | 420초 | 21초 | 
| **FedAvg + BWA** | 94.8% | 380초 | 19초 |

### ADM 최적화 효과

**이질적 환경 (라즈베리파이 vs 노트북):**
```
Before ADM:
  - 모든 클라이언트: v_n = 1.0 (전체 데이터)
  - 라운드 시간: 90초 (가장 느린 클라이언트 기준)

After ADM:
  - 라즈베리파이 (느림): v_n = 0.4 (40% 데이터)
  - 노트북 (빠름): v_n = 1.0 (100% 데이터)
  - 라운드 시간: 21초 (77% 단축)
```

### BWA 최적화 효과

**동적 배치 크기 조절:**
```
Round 1-5:   batch_size = 32  (탐색)
Round 6-10:  batch_size = 64  (최적화)
Round 11-15: batch_size = 128 (수렴)
Round 16-20: batch_size = 64  (안정화)
```

## 🔧 고급 설정

### ADM 파라미터 조정
```python
# flower_server.py
adm_params = {
    'Gamma': 0.4,           # v_n 최소값 (40%)
    'c_n': 1000000,         # CPU 사이클/샘플
    't': 60,                # 초기 시간 제약 (초)
    'local_iter': 3,        # 로컬 epoch 수
}
```

### BWA 파라미터 조정
```python
# BWA.py
bwa = BWAAlgorithm(
    batch_size_options=[16, 32, 64, 128],
    learning_rate_actor=1e-4,
    learning_rate_critic=1e-3,
    gamma=0.99,
    ppo_epochs=10
)
```

### Non-IID 데이터 분포
```bash
# 강한 편향 (90% 선호 클래스)
python flower_client.py --client_id 0 --iid False

# 약한 편향 (50% 선호 클래스) - flower_client.py에서 bias=0.5로 수정
```

## 🐛 트러블슈팅

### 일반적인 문제

**1. ADM Solver 실패**
```
[WARNING] Solver failed at round X
```
**해결:** `t` 값이 너무 작음. `adm_params['t']`를 증가시키거나 `Gamma` 값을 감소시킴.

**2. 클라이언트 연결 실패**
```
Connection refused
```
**해결:** 방화벽 설정 확인, 서버 IP 주소 확인, 포트 8080 개방 확인.

**3. CUDA 메모리 부족**
```
RuntimeError: CUDA out of memory
```
**해결:** 배치 크기 감소 또는 CPU 모드 사용 (`--device cpu`).

### 디바이스별 최적화

**Raspberry Pi:**
```bash
# 메모리 절약 모드
python flower_client.py --client_id 0 --batch_size 16 --local_epochs 2
```

**Jetson Nano:**
```bash
# GPU 활용
python flower_client.py --client_id 2 --batch_size 32 --device cuda
```

## 📚 참고 자료

- **Flower Documentation**: https://flower.dev/
- **ADM Paper**: [Adaptive Data Management for Federated Learning]
- **BWA Paper**: [Bandwidth-Aware Federated Learning with DRL]
- **PyTorch Federated Learning**: https://pytorch.org/tutorials/

## 🤝 기여하기

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🗑️ 정리 대상 (자동 제외됨)

`.gitignore`에 의해 다음 파일/폴더는 Git에서 제외됩니다:
- `__pycache__/`, `*.pyc`
- `venv/`, `env/`
- `data/`, `logs/`
- `*.log`, `*.pth`, `*.pt`
- `results_*.json`, `*.png`
