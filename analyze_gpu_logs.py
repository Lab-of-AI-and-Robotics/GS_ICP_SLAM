# analyze_gpu_logs.py
# 여러 scene의 GPU 로그를 통합 분석하고 비교하는 스크립트

import pandas as pd  # 데이터 분석
import matplotlib.pyplot as plt  # 시각화
import numpy as np  # 수치 연산
from pathlib import Path  # 경로 처리


def analyze_all_gpu_logs(results_dir="/home/GS_ICP_SLAM/texture_analysis_results"):
    """
    모든 scene의 GPU 로그를 통합 분석하는 함수

    Args:
        results_dir: GPU 로그 파일들이 저장된 디렉토리

    Returns:
        list: 각 scene의 GPU 사용 통계 리스트

    기능:
        1. 모든 gpu_log_*.csv 파일 찾기
        2. 각 로그 파일 읽고 통계 계산
        3. 비교 시각화
    """
    results_path = Path(results_dir)

    # gpu_log_로 시작하는 모든 CSV 파일 찾기
    # glob: 파일 패턴 매칭
    log_files = list(results_path.glob("gpu_log_*.csv"))

    if not log_files:
        print("GPU 로그 파일을 찾을 수 없습니다.")
        return

    print(f"{len(log_files)}개의 GPU 로그 분석 중...")

    # 각 scene의 통계를 저장할 리스트
    all_stats = []

    # 각 로그 파일 처리
    for log_file in sorted(log_files):
        # 파일명에서 scene 이름 추출
        # 예: gpu_log_room0.csv -> room0
        scene_name = log_file.stem.replace("gpu_log_", "")

        try:
            # CSV 파일 읽기
            df = pd.read_csv(log_file)

            # 통계 계산
            stats = {
                "scene": scene_name,
                "avg_memory_used": df["memory_used"].mean(),  # 평균 메모리 사용량
                "max_memory_used": df["memory_used"].max(),  # 최대 메모리 사용량
                "avg_gpu_util": df["gpu_util"].mean(),  # 평균 GPU 사용률
                "max_gpu_util": df["gpu_util"].max(),  # 최대 GPU 사용률
                "duration": len(df),  # 실행 시간(초)
            }

            all_stats.append(stats)

            # 콘솔에 출력
            print(f"\n{scene_name}:")
            print(f"  평균 메모리: {stats['avg_memory_used']:.0f} MB")
            print(f"  최대 메모리: {stats['max_memory_used']:.0f} MB")
            print(f"  평균 GPU 사용률: {stats['avg_gpu_util']:.1f}%")

        except Exception as e:
            print(f"{log_file} 분석 실패: {e}")

    # 시각화 함수 호출
    visualize_gpu_stats(all_stats, results_dir)

    return all_stats


def visualize_gpu_stats(stats_list, output_dir):
    """
    GPU 통계를 시각화하는 함수

    Args:
        stats_list: 각 scene의 GPU 통계 리스트
        output_dir: 그래프를 저장할 디렉토리

    생성되는 그래프:
        1. GPU 메모리 사용량 비교 (평균 vs 최대)
        2. GPU 사용률 비교
    """
    # 데이터 추출
    scenes = [s["scene"] for s in stats_list]
    avg_mem = [s["avg_memory_used"] for s in stats_list]
    max_mem = [s["max_memory_used"] for s in stats_list]
    avg_util = [s["avg_gpu_util"] for s in stats_list]

    # 2개의 서브플롯 생성 (2행 1열)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # --- 그래프 1: GPU 메모리 사용량 ---
    # 평균과 최대값을 나란히 비교
    x = np.arange(len(scenes))  # scene 위치
    width = 0.35  # 막대 너비

    # 평균 메모리 (파란색)
    ax1.bar(x - width / 2, avg_mem, width, label="Average", color="blue", alpha=0.7)
    # 최대 메모리 (빨간색)
    ax1.bar(x + width / 2, max_mem, width, label="Maximum", color="red", alpha=0.7)

    ax1.set_xlabel("Scene", fontsize=12)
    ax1.set_ylabel("Memory (MB)", fontsize=12)
    ax1.set_title("GPU Memory Usage by Scene", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenes, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # --- 그래프 2: GPU 사용률 ---
    ax2.bar(scenes, avg_util, color="green", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Scene", fontsize=12)
    ax2.set_ylabel("Utilization (%)", fontsize=12)
    ax2.set_title("Average GPU Utilization by Scene", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 레이아웃 조정
    plt.tight_layout()

    # 파일로 저장
    # dpi=300: 고해상도
    # bbox_inches='tight': 여백 최소화
    plt.savefig(
        f"{output_dir}/gpu_analysis_comparison.png", dpi=300, bbox_inches="tight"
    )
    print(f"\nGPU 분석 그래프 저장: {output_dir}/gpu_analysis_comparison.png")


# 메인 실행
if __name__ == "__main__":
    # GPU 로그 분석 시작
    analyze_all_gpu_logs()
