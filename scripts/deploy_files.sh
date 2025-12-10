#!/bin/bash
# Federated Learning 파일 배포 스크립트
# 각 디바이스에 필요한 파일만 전송

echo "=========================================="
echo "FL Files Deployment Script"
echo "=========================================="

# 서버 파일 리스트
SERVER_FILES=(
    "flower_server.py"
    "models.py"
    "utils.py"
    "ADM.py"
    "compare_strategies.py"
    "requirements.txt"
    "check_versions.py"
    "dists.py"
)

# 클라이언트 파일 리스트
CLIENT_FILES=(
    "flower_client.py"
    "models.py"
    "utils.py"
    "updateModel.py"
    "dists.py"
    "requirements.txt"
    "check_versions.py"
)

# 설정: 디바이스 IP 주소 (여기를 수정하세요)
RPI_IP="192.168.0.102"
RPI_USER="pi"
RPI_PATH="~/fl/"

# 함수: 파일 전송
deploy_client() {
    local host=$1
    local user=$2
    local path=$3
    local device=$4

    echo ""
    echo "Deploying to $device ($user@$host)..."

    # 디렉토리 생성
    ssh $user@$host "mkdir -p $path"

    # 파일 전송
    for file in "${CLIENT_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo "  → $file"
            scp "$file" $user@$host:$path
        else
            echo "  ✗ $file not found!"
        fi
    done

    echo "✓ Deployment to $device completed!"
}

# 메뉴
echo ""
echo "Select deployment option:"
echo "  1) Deploy to Raspberry Pi"
echo "  2) Check server files (this machine)"
echo ""
read -p "Enter choice [1-2]: " choice

case $choice in
    1)
        deploy_client $RPI_IP $RPI_USER $RPI_PATH "Raspberry Pi"
        ;;
    2)
        echo ""
        echo "Checking server files on this machine..."
        echo ""
        for file in "${SERVER_FILES[@]}"; do
            if [ -f "$file" ]; then
                echo "  ✓ $file"
            else
                echo "  ✗ $file (missing)"
            fi
        done
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Deployment completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. SSH to each device"
echo "  2. Run: python3 check_versions.py"
echo "  3. Run: python3 flower_client.py --client_id X --server_address <SERVER_IP>:8080"
echo ""
