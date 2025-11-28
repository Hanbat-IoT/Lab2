# 빠른 시작 가이드

## 🚀 5분 안에 FL 시스템 구축하기

---

## 📋 준비물

1. **서버** (노트북/데스크톱)
2. **클라이언트 1** (Jetson Nano 또는 Raspberry Pi)
3. **클라이언트 2** (Jetson Nano 또는 Raspberry Pi)
4. 같은 네트워크에 연결

---

## 1️⃣ 서버 설정 (노트북)

```bash
# 1. 의존성 설치
pip install -r configs/requirements.txt

# 2. 서버 IP 확인
hostname -I
# 예: 192.168.0.100

# 3. 서버 실행
python flower_server.py \
  --strategy fedavg_adm \
  --num_clients 2 \
  --num_rounds 10 \
  --dataset mnist

# 서버가 클라이언트를 기다립니다...
```

---

## 2️⃣ Jetson Nano 설정

### Docker 사용 (추천)
```bash
# 1. 이미지 빌드
cd configs
docker build -f Dockerfile.jetson -t fl-client-jetson ..

# 2. 실행 (서버 IP 변경 필요)
docker run --runtime nvidia --network host \
  -e CLIENT_ID=0 \
  -e SERVER_ADDRESS=192.168.0.100:8080 \
  fl-client-jetson:latest
```

### 직접 설치
```bash
# 1. 자동 설치
bash configs/install-jetson.sh

# 2. 실행 (서버 IP 변경 필요)
python3 flower_client_jetson_py36.py \
  --client_id 0 \
  --server_address 192.168.0.100:8080 \
  --data_size 1500
```

---

## 3️⃣ Raspberry Pi 설정

### Docker 사용 (추천)
```bash
# 1. 이미지 빌드
cd configs
docker build -f Dockerfile.rpi -t fl-client-rpi ..

# 2. 실행 (서버 IP 변경 필요)
docker run --network host \
  -e CLIENT_ID=1 \
  -e SERVER_ADDRESS=192.168.0.100:8080 \
  fl-client-rpi:latest
```

### 직접 설치
```bash
# 1. 자동 설치
bash configs/install-rpi.sh

# 2. 실행 (서버 IP 변경 필요)
python3 flower_client.py \
  --client_id 1 \
  --server_address 192.168.0.100:8080 \
  --data_size 2000
```

---

## 📊 결과 확인

서버 터미널에서 학습 진행 상황을 확인할 수 있습니다:

```
==================================================
Round 1/10
==================================================
[ADM Optimization]
Optimized v_n: [0.85, 0.92]
...
Round 1 - Global Accuracy: 85.23%
==================================================
```

---

## 🔧 문제 해결

### 클라이언트가 서버에 연결 안됨
```bash
# 서버에서 방화벽 포트 열기
sudo ufw allow 8080/tcp

# 연결 테스트
ping <SERVER_IP>
telnet <SERVER_IP> 8080
```

### Jetson Nano 메모리 부족
```bash
# 데이터 크기 줄이기
--data_size 1000

# swap 메모리 증가
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Raspberry Pi 느린 학습
```bash
# 정상입니다! CPU만 사용하므로 느립니다.
# 데이터 크기를 줄이면 빨라집니다:
--data_size 1000
```

---

## 📝 다음 단계

1. **더 많은 클라이언트 추가**
   ```bash
   python flower_server.py --num_clients 5 --num_rounds 20
   ```

2. **다른 데이터셋 사용**
   ```bash
   --dataset cifar
   ```

3. **GUI 모드 사용**
   ```bash
   python app.py
   # http://localhost:8080 접속
   ```

4. **결과 비교**
   ```bash
   python scripts/compare_strategies.py \
     --baseline results_fedavg.json \
     --adm results_fedavg_adm.json
   ```

---

## 💡 팁

- **Jetson Nano**: `--data_size 1500` 권장
- **Raspberry Pi**: `--data_size 2000` 권장
- **노트북**: `--data_size 2500` 권장
- **메모리 부족 시**: 데이터 크기를 1000으로 줄이기
- **빠른 테스트**: `--num_rounds 5`로 시작

---

## 📚 더 자세한 정보

- **전체 설치 가이드**: `configs/DEVICE_SETUP.md`
- **Docker 가이드**: `configs/README_DOCKER.md`
- **호환성 노트**: `configs/COMPATIBILITY_NOTES.md`
- **Flower 가이드**: `docs/FLOWER_GUIDE.md`
