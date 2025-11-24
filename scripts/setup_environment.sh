#!/bin/bash
# Federated Learning Environment Setup Script
# 모든 디바이스에서 실행하여 동일한 환경 구축

echo "=========================================="
echo "FL Environment Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Detect device type
if [ -f "/etc/nv_tegra_release" ]; then
    DEVICE="jetson"
    echo "Device: NVIDIA Jetson"
elif grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    DEVICE="raspberry_pi"
    echo "Device: Raspberry Pi"
else
    DEVICE="laptop"
    echo "Device: Laptop/Desktop"
fi

# Create virtual environment (recommended)
echo ""
echo "Creating virtual environment..."
python3 -m venv fl_env
source fl_env/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies based on device
echo ""
echo "Installing dependencies..."

if [ "$DEVICE" = "jetson" ]; then
    echo "Installing for Jetson Nano (with CUDA support)..."
    # PyTorch for Jetson (pre-built wheel)
    pip install torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v50

elif [ "$DEVICE" = "raspberry_pi" ]; then
    echo "Installing for Raspberry Pi (CPU only)..."
    # Lightweight versions for Raspberry Pi
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

else
    echo "Installing for Laptop/Desktop..."
    # Standard installation
    pip install torch torchvision
fi

# Install other dependencies
pip install flwr==1.11.1
pip install cvxpy clarabel scs ecos
pip install numpy matplotlib pandas tqdm psutil

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python3 << EOF
import sys
try:
    import flwr
    print(f"✓ Flower: {flwr.__version__}")
except ImportError as e:
    print(f"✗ Flower: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")
    sys.exit(1)

try:
    import cvxpy
    print(f"✓ CVXPY: {cvxpy.__version__}")
except ImportError as e:
    print(f"✗ CVXPY: {e}")
    sys.exit(1)

print("\n✓ All dependencies installed successfully!")
EOF

echo ""
echo "=========================================="
echo "Setup completed!"
echo "=========================================="
echo ""
echo "To activate this environment in the future:"
echo "  source fl_env/bin/activate"
echo ""
