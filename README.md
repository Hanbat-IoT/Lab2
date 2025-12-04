# 프로젝트 파일 구조

## 📁 디렉토리 구조

```
FL_GUI/
│
├── docs/                           # 📚 문서
│   ├── README.md                   # 프로젝트 전체 설명
│   └── FLOWER_GUIDE.md            # Flower 네트워크 FL 가이드
│
├── configs/                        # ⚙️ 설정 파일
│   └── requirements.txt           # Python 의존성
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

### 클라이언트
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

## 🗑️ 정리 대상 (자동 제외됨)

`.gitignore`에 의해 다음 파일/폴더는 Git에서 제외됩니다:
- `__pycache__/`, `*.pyc`
- `venv/`, `env/`
- `data/`, `logs/`
- `*.log`, `*.pth`, `*.pt`
- `results_*.json`, `*.png`
