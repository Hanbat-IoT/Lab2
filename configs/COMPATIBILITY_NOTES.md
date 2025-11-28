# Jetson Nano Python 3.6 호환성 노트

## 🔴 중요: Flower 버전 차이

### Flower 1.4.0 (Python 3.6) vs 1.11.1 (Python 3.9+)

Jetson Nano (Python 3.6.9)에서는 Flower 1.4.0을 사용해야 하며, 일부 API가 다릅니다.

## 📝 코드 수정 필요 사항

### 1. flower_client.py 수정 (Python 3.6 호환)

**문제점**:
- Flower 1.4.0은 `NumPyClient` 대신 `Client` 사용
- `get_parameters()`, `set_parameters()` 시그니처 다름
- `config` 파라미터 처리 방식 다름

**해결 방법**:

```python
# Jetson Nano (Python 3.6, Flower 1.4.0)용 flower_client.py 수정

import sys
import flwr as fl

# Flower 버전 확인
FLOWER_VERSION = tuple(map(int, fl.__version__.split('.')[:2]))
USE_OLD_API = FLOWER_VERSION < (1, 5)

if USE_OLD_API:
    # Flower 1.4.0 (Python 3.6)
    from flwr.client import Client as FlowerClientBase
else:
    # Flower 1.5+ (Python 3.9+)
    from flwr.client import NumPyClient as FlowerClientBase

class FlowerClient(FlowerClientBase):
    def __init__(self, client_id, dataset, data_size=2500):
        # ... 기존 코드 ...
        pass
    
    if USE_OLD_API:
        # Flower 1.4.0 API
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
        def fit(self, parameters, config):
            self.set_parameters(parameters)
            # ... 학습 코드 ...
            return self.get_parameters(), len(adjusted_data), {}
        
        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            # ... 평가 코드 ...
            return float(0.0), len(self.testset), {"accuracy": float(accuracy)}
    else:
        # Flower 1.5+ API (기존 코드)
        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
        def fit(self, parameters, config):
            # ... 기존 코드 ...
            pass
        
        def evaluate(self, parameters, config):
            # ... 기존 코드 ...
            pass
```

### 2. 타입 힌트 제거 (Python 3.6 호환)

Python 3.6은 일부 타입 힌트를 지원하지 않습니다.

```python
# 수정 전 (Python 3.9+)
from typing import List, Tuple, Dict, Optional

def function(data: List[int]) -> Dict[str, float]:
    pass

# 수정 후 (Python 3.6)
from typing import List, Tuple, Dict, Optional

def function(data):
    # type: (List[int]) -> Dict[str, float]
    pass
```

### 3. f-string 사용 가능

Python 3.6부터 f-string 지원하므로 그대로 사용 가능:
```python
print(f"Client {client_id}: accuracy = {accuracy:.2f}")  # OK
```

### 4. dataclasses 백포트 필요

```bash
pip3 install dataclasses==0.8
```

## 🔧 간단한 해결책: 조건부 import

`flower_client.py` 상단에 추가:

```python
import sys

# Python 버전 확인
if sys.version_info < (3, 7):
    print("Warning: Python 3.6 detected. Using Flower 1.4.0 compatible mode.")
    # Python 3.6 호환 모드
    import dataclasses  # 백포트 필요
```

## 🚀 권장 사항

### 옵션 1: Flower 1.4.0 전용 클라이언트 파일 생성 (권장)

```bash
# Jetson Nano용 별도 파일
cp flower_client.py flower_client_jetson_py36.py
# 수정 후 사용
python3 flower_client_jetson_py36.py --client_id 0 --server_address <IP>:8080
```

### 옵션 2: 조건부 코드로 통합

기존 `flower_client.py`에 버전 체크 로직 추가하여 양쪽 모두 지원

### 옵션 3: JetPack 5.x로 업그레이드 (최선)

JetPack 5.x는 Python 3.8+를 지원하므로 최신 라이브러리 사용 가능

## 📦 전체 의존성 요약

### Jetson Nano (JetPack 4.x, Python 3.6.9)
```
flwr==1.4.0
torch==1.10.0 (JetPack 포함)
numpy==1.19.5
cvxpy==1.1.18
scs==2.1.4
ecos==2.0.10
matplotlib==3.3.4
pandas==1.1.5
tqdm==4.62.3
psutil==5.8.0
typing-extensions==4.1.1
dataclasses==0.8
```

### Raspberry Pi / 노트북 (Python 3.9+)
```
flwr==1.11.1
torch==2.0.0
numpy==1.24.3
cvxpy==1.3.2
scs==3.2.3
ecos==2.0.12
matplotlib==3.7.2
pandas==2.0.3
tqdm==4.66.1
psutil==5.9.5
```

## ⚠️ 알려진 제한사항

1. **Flower 1.4.0 제한**:
   - 최신 전략 (FedProx, FedOpt 등) 일부 미지원
   - gRPC 버전 제한
   - 일부 메트릭 로깅 기능 제한

2. **Python 3.6 제한**:
   - `typing` 모듈 일부 기능 미지원
   - `dataclasses` 백포트 필요
   - 일부 최신 문법 미지원

3. **PyTorch 1.10 제한**:
   - 일부 최신 연산자 미지원
   - TorchScript 기능 제한

## 🔍 테스트 방법

```bash
# Jetson Nano에서 버전 확인
python3 --version
python3 -c "import flwr; print(flwr.__version__)"
python3 -c "import torch; print(torch.__version__)"

# 간단한 연결 테스트
python3 flower_client.py --client_id 0 --server_address <IP>:8080 --dataset mnist
```
