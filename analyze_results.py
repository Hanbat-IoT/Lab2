"""
간단한 FL 결과 분석 스크립트

사용법:
  python analyze_results.py results_fedavg_adm_1clients.json
"""

import json
import sys
import matplotlib.pyplot as plt


def analyze_results(filename):
    """결과 파일 분석"""
    
    # JSON 파일 로드
    with open(filename, 'r') as f:
        results = json.load(f)
    
    print("=" * 70)
    print("📊 Federated Learning 결과 분석")
    print("=" * 70)
    
    # 기본 정보
    print(f"\n전략: {results['strategy']}")
    print(f"라운드 수: {results['num_rounds']}")
    
    # 정확도 분석
    accuracies = results['accuracies']
    print(f"\n📈 정확도:")
    print(f"  최종: {accuracies[-1]*100:.2f}%")
    print(f"  최고: {max(accuracies)*100:.2f}%")
    print(f"  평균: {sum(accuracies)/len(accuracies)*100:.2f}%")
    
    # 라운드별 정확도
    print(f"\n📋 라운드별 정확도:")
    for i, acc in enumerate(accuracies, 1):
        print(f"  Round {i}: {acc*100:.2f}%")
    
    # v_n 히스토리 (ADM인 경우)
    if 'v_n_history' in results and results['v_n_history']:
        print(f"\n🔧 ADM 데이터 사용량 (v_n):")
        for i, v_n_list in enumerate(results['v_n_history'], 1):
            v_n_str = ", ".join([f"{v:.3f}" for v in v_n_list])
            print(f"  Round {i}: [{v_n_str}]")
    
    # 간단한 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies)+1), [a*100 for a in accuracies], 
             marker='o', linewidth=2, markersize=8)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'{results["strategy"]} - Accuracy over Rounds', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = filename.replace('.json', '_plot.png')
    plt.savefig(output_file, dpi=300)
    print(f"\n📊 그래프 저장: {output_file}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python analyze_results.py <results.json>")
        sys.exit(1)
    
    analyze_results(sys.argv[1])
