# Docker 사용 가이드

## 🐳 Docker를 사용한 FL 클라이언트 배포

Docker를 사용하면 의존성 설치 없이 바로 실행할 수 있습니다.

---

## 📦 사전 준비

### Jetson Nano
```bash
# Docker 설치 확인
docker --version

# NVIDIA Container Runtime 설치 (GPU 사용)
sudo apt-get install -y nvidia-container-runtime

# Docker에 NVIDIA runtime 추가
sudo nano /etc/docker/daemon.json
# 다음 내용 추가:
# {
#   "runtimes": {
#     "nvidia": {
#       "path": "nvidia-container-runtime",
#       "runtimeArgs": []
#     }
#   }
# }

sudo systemctl restart docker
```

### Raspberry Pi
```bash
# Docker 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 재로그인 필요
exit
```

---

## 🚀 빠른 시작

### 1. 이미지 빌드

#### Jetson Nano
```bash
cd configs
docker build -f Dockerfile.jetson -t fl-client-jetson:latest ..
```

#### Raspberry Pi
```bash
cd configs
docker build -f Dockerfile.rpi -t fl-client-rpi:latest ..
```

### 2. 컨테이너 실행

#### Jetson Nano
```bash
docker run --runtime nvidia --network host \
  -e CLIENT_ID=0 \
  -e SERVER_ADDRESS=192.168.0.100:8080 \
  -e DATASET=mnist \
  -e DATA_SIZE=1500 \
  fl-client-jetson:latest
```

#### Raspberry Pi
```bash
docker run --network host \
  -e CLIENT_ID=1 \
  -e SERVER_ADDRESS=192.168.0.100:8080 \
  -e DATASET=cifar \
  -e DATA_SIZE=2000 \
  fl-client-rpi:latest
```

---

## 🔧 Docker Compose 사용 (권장)

### 1. 환경 변수 설정
```bash
# configs/.env 파일 생성
cat > configs/.env << EOF
# Jetson Nano
JETSON_CLIENT_ID=0
JETSON_SERVER_ADDRESS=192.168.0.100:8080
JETSON_DATASET=mnist
JETSON_DATA_SIZE=1500

# Raspberry Pi
RPI_CLIENT_ID=1
RPI_SERVER_ADDRESS=192.168.0.100:8080
RPI_DATASET=mnist
RPI_DATA_SIZE=2000
EOF
```

### 2. 실행
```bash
# Jetson Nano에서
cd configs
docker-compose up jetson-client

# Raspberry Pi에서
cd configs
docker-compose up rpi-client

# 백그라운드 실행
docker-compose up -d jetson-client
```

### 3. 관리
```bash
# 로그 확인
docker-compose logs -f jetson-client

# 중지
docker-compose stop

# 재시작
docker-compose restart

# 삭제
docker-compose down
```

---

## 📝 Dockerfile 설명

### Jetson Nano (Dockerfile.jetson)
- **베이스 이미지**: `nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3`
  - JetPack 4.x, Python 3.6.9, PyTorch 1.10.0 포함
- **의존성**: Flower 1.4.0 (Python 3.6 호환)
- **클라이언트**: `flower_client_jetson_py36.py` 사용

### Raspberry Pi (Dockerfile.rpi)
- **베이스 이미지**: `python:3.9-slim-bullseye`
  - Python 3.9, ARM64 최적화
- **의존성**: Flower 1.11.1, PyTorch 2.0.0 (CPU)
- **클라이언트**: `flower_client.py` 사용

---

## 🔍 문제 해결

### Jetson Nano: NVIDIA runtime 오류
```bash
# 오류: docker: Error response from daemon: Unknown runtime specified nvidia
# 해결:
sudo apt-get install -y nvidia-container-runtime
sudo systemctl restart docker
```

### 이미지 빌드 실패
```bash
# 캐시 없이 재빌드
docker build --no-cache -f Dockerfile.jetson -t fl-client-jetson:latest ..
```

### 네트워크 연결 오류
```bash
# host 네트워크 모드 사용 (권장)
docker run --network host ...

# 또는 포트 포워딩
docker run -p 8080:8080 ...
```

### 메모리 부족
```bash
# 메모리 제한 설정
docker run --memory="2g" --memory-swap="4g" ...
```

---

## 📊 이미지 크기 비교

| 이미지 | 크기 | 설명 |
|--------|------|------|
| fl-client-jetson | ~5GB | NVIDIA L4T + PyTorch + CUDA |
| fl-client-rpi | ~2GB | Python 3.9 + PyTorch CPU |

---

## 🧹 정리

```bash
# 컨테이너 중지 및 삭제
docker stop fl-jetson-client fl-rpi-client
docker rm fl-jetson-client fl-rpi-client

# 이미지 삭제
docker rmi fl-client-jetson:latest fl-client-rpi:latest

# 사용하지 않는 이미지 정리
docker system prune -a
```

---

## 💡 팁

1. **개발 중**: 볼륨 마운트로 코드 수정 반영
   ```bash
   docker run -v $(pwd):/app ...
   ```

2. **로그 저장**: 로그 디렉토리 마운트
   ```bash
   docker run -v $(pwd)/logs:/app/logs ...
   ```

3. **자동 재시작**: `--restart unless-stopped` 옵션 사용
   ```bash
   docker run --restart unless-stopped ...
   ```

4. **멀티 클라이언트**: 같은 디바이스에서 여러 클라이언트 실행
   ```bash
   docker run --name client0 -e CLIENT_ID=0 ...
   docker run --name client1 -e CLIENT_ID=1 ...
   ```
