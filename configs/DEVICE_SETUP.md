# 디바이스별 설치 가이드

## 📋 의존성 버전 정보

### 노트북/서버 & Raspberry Pi (Python 3.9+)
- **Flower**: 1.11.1
- **PyTorch**: 2.0.0
- **NumPy**: 1.24.3
- **CVXPY**: 1.3.2
- **SCS**: 3.2.3
- **ECOS**: 2.0.12
- **Matplotlib**: 3.7.2
- **Pandas**: 2.0.3
- **tqdm**: 4.66.1
- **psutil**: 5.9.5

### Jetson Nano (Python 3.6.9 - JetPack 4.x)
- **Flower**: 1.4.0 (Python 3.6 호환)
- **PyTorch**: 1.10.0 (JetPack 포함, 설치 불필요)
- **NumPy**: 1.19.5
- **CVXPY**: 1.1.18
- **SCS**: 2.1.4
- **ECOS**: 2.0.10
- **Matplotlib**: 3.3.4
- **Pandas**: 1.1.5
- **tqdm**: 4.62.3
- **psutil**: 5.8.0
- **typing-extensions**: 4.1.1
- **dataclasses**: 0.8

### Jetson Nano (Python 3.8+ - JetPack 5.x)
- 노트북/서버와 동일한 버전 사용 가능

---

## 🖥️ 노트북/서버 (Windows/Linux/Mac)

### 설치 방법
```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r configs/requirements.txt

# GPU 사용 시 (CUDA 11.8)
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118
```

### 실행
```bash
# 서버 실행
python flower_server.py --strategy fedavg_adm --num_clients 3 --num_rounds 20

# 또는 GUI 모드
python app.py
```

---

## 🤖 Jetson Nano

### 사전 요구사항
- **JetPack**: 4.6.x (Python 3.6.9) 또는 5.x (Python 3.8+)
- **Python**: 3.6.9 (JetPack 4.x) / 3.8+ (JetPack 5.x)
- **CUDA**: 10.2 (JetPack 4.x) / 11.4 (JetPack 5.x)

### ⚠️ 중요: Python 버전 확인
```bash
python3 --version
# Python 3.6.9 → JetPack 4.x 사용 중
# Python 3.8+  → JetPack 5.x 사용 중
```

### 방법 1: Docker 사용 (권장)

```bash
# Docker 이미지 빌드
cd configs
docker build -f Dockerfile.jetson -t fl-client-jetson ..

# 컨테이너 실행
docker run --runtime nvidia --network host \
  fl-client-jetson \
  python3 flower_client.py \
  --client_id 0 \
  --server_address <SERVER_IP>:8080 \
  --dataset mnist
```

### 방법 2: 자동 설치 스크립트 (권장)

```bash
# 설치 스크립트 실행 (Python 버전 자동 감지)
bash configs/install-jetson.sh

# 클라이언트 실행
# Python 3.6.9인 경우:
python3 flower_client_jetson_py36.py \
  --client_id 0 \
  --server_address <SERVER_IP>:8080 \
  --dataset mnist \
  --data_size 1500

# Python 3.8+인 경우:
python3 flower_client.py \
  --client_id 0 \
  --server_address <SERVER_IP>:8080 \
  --dataset mnist \
  --data_size 1500
```

### 방법 3: 수동 설치 (Python 3.6.9 - JetPack 4.x)

```bash
# Python 버전 확인
python3 --version  # Python 3.6.9 확인

# 시스템 패키지 설치
sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev python3-pip

# pip 업그레이드 (Python 3.6 호환 버전)
python3 -m pip install --upgrade pip==21.3.1

# PyTorch는 이미 JetPack에 포함되어 있음 (1.10.0)
# Python 3.6 호환 의존성 설치
pip3 install -r configs/requirements-jetson-py36.txt

# 또는 개별 설치:
pip3 install \
  typing-extensions==4.1.1 \
  dataclasses==0.8 \
  numpy==1.19.5 \
  flwr==1.4.0 \
  cvxpy==1.1.18 \
  scs==2.1.4 \
  ecos==2.0.10 \
  matplotlib==3.3.4 \
  pandas==1.1.5 \
  tqdm==4.62.3 \
  psutil==5.8.0

# 클라이언트 실행 (Python 3.6 전용 파일 사용)
python3 flower_client_jetson_py36.py \
  --client_id 0 \
  --server_address <SERVER_IP>:8080 \
  --dataset mnist \
  --data_size 1500
```

### 방법 4: 수동 설치 (Python 3.8+ - JetPack 5.x)

```bash
# Python 버전 확인
python3 --version  # Python 3.8+ 확인

# 시스템 패키지 설치
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-pip

# 최신 버전 의존성 설치
pip3 install --upgrade pip
pip3 install \
  flwr==1.11.1 \
  numpy==1.24.3 \
  cvxpy==1.3.2 \
  scs==3.2.3 \
  ecos==2.0.12 \
  matplotlib==3.7.2 \
  pandas==2.0.3 \
  tqdm==4.66.1 \
  psutil==5.9.5

# 클라이언트 실행
python3 flower_client.py \
  --client_id 0 \
  --server_address <SERVER_IP>:8080 \
  --dataset mnist
```

### 주의사항
- **Python 3.6.9 사용 시**: Flower 1.4.0 버전 사용 (최신 버전 호환 안됨)
- Jetson Nano는 메모리가 제한적이므로 `--data_size 1500` 옵션 사용 권장
- CUDA 메모리 부족 시 배치 사이즈 줄이기: `batch_size=16`
- JetPack 4.x는 PyTorch 1.10.0 포함 (별도 설치 불필요)
- JetPack 5.x는 PyTorch 2.0 포함

---

## 🍓 Raspberry Pi (4/5)

### 사전 요구사항
- **OS**: Raspberry Pi OS (64-bit) 권장
- **Python**: 3.9+
- **RAM**: 4GB 이상 권장

### 방법 1: Docker 사용 (권장)

```bash
# Docker 설치 (아직 없다면)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 이미지 빌드
cd configs
docker build -f Dockerfile.rpi -t fl-client-rpi ..

# 컨테이너 실행
docker run --network host \
  fl-client-rpi \
  python3 flower_client.py \
  --client_id 1 \
  --server_address <SERVER_IP>:8080 \
  --dataset mnist
```

### 방법 2: 자동 설치 스크립트 (권장)

```bash
# 설치 스크립트 실행
bash configs/install-rpi.sh

# 클라이언트 실행
python3 flower_client.py \
  --client_id 1 \
  --server_address <SERVER_IP>:8080 \
  --dataset mnist \
  --data_size 2000
```

### 방법 3: 수동 설치

```bash
# 시스템 패키지 설치
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  git \
  libopenblas-dev \
  liblapack-dev \
  gfortran \
  python3-pip

# Python 의존성 설치
pip3 install --upgrade pip

# NumPy 먼저 설치 (ARM 최적화)
pip3 install numpy==1.24.3

# PyTorch CPU 버전 설치
pip3 install torch==2.0.0 torchvision==0.15.0 \
  --index-url https://download.pytorch.org/whl/cpu

# 나머지 의존성 설치
pip3 install -r configs/requirements-rpi.txt

# 또는 개별 설치:
pip3 install \
  flwr==1.11.1 \
  cvxpy==1.3.2 \
  scs==3.2.3 \
  ecos==2.0.12 \
  matplotlib==3.7.2 \
  pandas==2.0.3 \
  tqdm==4.66.1 \
  psutil==5.9.5

# 클라이언트 실행
python3 flower_client.py \
  --client_id 1 \
  --server_address <SERVER_IP>:8080 \
  --dataset mnist \
  --data_size 2000
```

### 주의사항
- 라즈베리파이는 CPU만 사용하므로 학습 속도가 느림
- 메모리 부족 시 `--data_size 1000` 사용
- swap 메모리 증가 권장:
  ```bash
  sudo dphys-swapfile swapoff
  sudo nano /etc/dphys-swapfile
  # CONF_SWAPSIZE=2048 로 변경
  sudo dphys-swapfile setup
  sudo dphys-swapfile swapon
  ```

---

## 🔧 문제 해결

### CVXPY 설치 오류 (ARM 디바이스)
```bash
# 시스템 라이브러리 먼저 설치
sudo apt-get install -y libopenblas-dev liblapack-dev

# Python 3.6 (Jetson Nano JetPack 4.x)
pip3 install cvxpy==1.1.18 scs==2.1.4

# Python 3.8+ (Raspberry Pi, JetPack 5.x)
pip3 install cvxpy==1.3.2 scs==3.2.3
```

### Flower 버전 오류 (Jetson Nano Python 3.6)
```bash
# Python 3.6은 Flower 1.5+ 지원 안함
pip3 install flwr==1.4.0

# typing-extensions 필요
pip3 install typing-extensions==4.1.1 dataclasses==0.8
```

### PyTorch 설치 오류 (Raspberry Pi)
```bash
# CPU 버전 명시적으로 설치
pip3 install torch==2.0.0 torchvision==0.15.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

### 메모리 부족 오류
```bash
# 데이터 크기 줄이기
python3 flower_client.py --data_size 1000

# 배치 사이즈 줄이기 (코드 수정 필요)
# flower_client.py에서 batch_size=16으로 변경
```

### 네트워크 연결 오류
```bash
# 서버 IP 확인
# 서버에서 실행:
hostname -I

# 방화벽 포트 열기 (서버)
sudo ufw allow 8080/tcp

# 연결 테스트
ping <SERVER_IP>
telnet <SERVER_IP> 8080
```

---

## 📊 성능 비교

| 디바이스 | CPU | RAM | Python | PyTorch | 학습 속도 | 권장 data_size |
|---------|-----|-----|--------|---------|----------|---------------|
| 노트북 (GPU) | i7 | 16GB | 3.9+ | 2.0+ | ~5.0 | 2500 |
| 노트북 (CPU) | i7 | 16GB | 3.9+ | 2.0+ | ~1.0 | 2500 |
| Jetson Nano (JP4) | ARM A57 | 4GB | 3.6.9 | 1.10 | ~0.8 | 1500 |
| Jetson Nano (JP5) | ARM A57 | 4GB | 3.8+ | 2.0 | ~1.0 | 1500 |
| Raspberry Pi 4 | ARM A72 | 4GB | 3.9+ | 2.0 | ~0.3 | 1000-2000 |
| Raspberry Pi 5 | ARM A76 | 8GB | 3.9+ | 2.0 | ~0.5 | 2000 |

---

## 🚀 빠른 테스트

### 서버 (노트북)
```bash
python flower_server.py --num_clients 2 --num_rounds 5 --dataset mnist
```

### 클라이언트 1 (Jetson Nano)
```bash
python3 flower_client.py --client_id 0 --server_address 192.168.0.100:8080
```

### 클라이언트 2 (Raspberry Pi)
```bash
python3 flower_client.py --client_id 1 --server_address 192.168.0.100:8080
```

---

## 📝 버전 확인

```bash
# 설치된 버전 확인
python scripts/check_versions.py
```
