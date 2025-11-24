"""
버전 확인 스크립트
모든 디바이스에서 실행하여 버전 일치 여부 확인
"""
import sys
import platform

def check_versions():
    print("=" * 70)
    print("Federated Learning Environment Check")
    print("=" * 70)

    # System info
    print(f"\n[System Info]")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Processor: {platform.processor()}")
    print(f"  Python: {sys.version.split()[0]}")

    # Required versions (target)
    target_versions = {
        'flwr': '1.11.1',
        'torch': '2.1.0',
        'cvxpy': '1.4.2'
    }

    # Check installed versions
    print(f"\n[Package Versions]")
    all_ok = True

    for package, target_version in target_versions.items():
        try:
            if package == 'flwr':
                import flwr
                current_version = flwr.__version__
            elif package == 'torch':
                import torch
                current_version = torch.__version__
                cuda_available = torch.cuda.is_available()
                cuda_version = torch.version.cuda if cuda_available else "N/A"
            elif package == 'cvxpy':
                import cvxpy
                current_version = cvxpy.__version__

            # Version match
            if current_version.startswith(target_version.split('.')[0]):  # Major version match
                status = "✓"
            else:
                status = "⚠"
                all_ok = False

            print(f"  {status} {package}: {current_version} (target: {target_version})")

            # PyTorch CUDA info
            if package == 'torch':
                print(f"      CUDA: {cuda_available} (version: {cuda_version})")

        except ImportError:
            print(f"  ✗ {package}: NOT INSTALLED")
            all_ok = False

    # Check optional packages
    print(f"\n[Optional Packages]")
    optional = ['numpy', 'matplotlib', 'pandas', 'tqdm', 'psutil']
    for package in optional:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {package}: {version}")
        except ImportError:
            print(f"  ✗ {package}: NOT INSTALLED")

    # Summary
    print(f"\n{'=' * 70}")
    if all_ok:
        print("✓ Environment OK - Ready for Federated Learning!")
    else:
        print("⚠ Version mismatch detected!")
        print("  Please run: pip install -r requirements.txt")
    print("=" * 70)

    return all_ok

if __name__ == "__main__":
    check_versions()
