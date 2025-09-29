# monitor_gpu_memory.py
# GPU 메모리 사용량을 실시간으로 모니터링하고 분석하는 스크립트

import subprocess  # 시스템 명령어 실행을 위한 모듈
import time  # 시간 관련 기능
import pandas as pd  # 데이터 분석을 위한 라이브러리
import matplotlib.pyplot as plt  # 그래프 시각화
from datetime import datetime  # 날짜/시간 처리


def monitor_gpu_realtime(duration_seconds=None, log_file="gpu_memory_log.csv"):
    """
    실시간 GPU 메모리 모니터링 함수

    Args:
        duration_seconds: 모니터링 지속 시간(초). None이면 무한 실행
        log_file: 로그를 저장할 CSV 파일 이름

    동작 방식:
        1. nvidia-smi 명령어로 GPU 정보 수집
        2. 1초마다 메모리 사용량, GPU 사용률 등을 기록
        3. CSV 파일에 실시간으로 저장
    """
    print("GPU 메모리 모니터링 시작...")

    # 시작 시간 기록
    start_time = time.time()

    try:
        while True:
            # nvidia-smi 명령어 실행
            # --query-gpu: 가져올 정보 지정
            # --format: 출력 형식 (csv, 헤더 없음, 단위 제거)
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,memory.used,memory.free,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )

            # 명령어 실행 성공시
            if result.returncode == 0:
                output = result.stdout.strip()
                print(f"현재 상태: {output}")

                # CSV 파일에 추가 모드로 저장
                with open(log_file, "a") as f:
                    f.write(output + "\n")

            # 1초 대기
            time.sleep(1)

            # 지정된 시간이 경과하면 종료
            if duration_seconds and (time.time() - start_time) > duration_seconds:
                break

    except KeyboardInterrupt:
        # Ctrl+C로 중단시
        print("\n모니터링 중단")


def analyze_gpu_log(log_file="gpu_memory_log.csv"):
    """
    저장된 GPU 로그 파일을 분석하고 시각화하는 함수

    Args:
        log_file: 분석할 CSV 로그 파일 경로

    Returns:
        DataFrame: 분석된 데이터프레임 (성공시)
        None: 분석 실패시

    기능:
        1. CSV 파일 읽기
        2. 통계 계산 (평균, 최대, 최소)
        3. 그래프 생성 및 저장
    """
    try:
        # CSV 파일 읽기
        # names: 컬럼 이름 지정
        df = pd.read_csv(
            log_file,
            names=[
                "timestamp",
                "memory_used",
                "memory_free",
                "memory_total",
                "gpu_util",
            ],
        )

        # 통계 계산 및 출력
        print("\nGPU 메모리 사용 통계:")
        print(f"  평균 사용량: {df['memory_used'].mean():.2f} MB")
        print(f"  최대 사용량: {df['memory_used'].max():.2f} MB")
        print(f"  최소 사용량: {df['memory_used'].min():.2f} MB")
        print(f"  평균 GPU 사용률: {df['gpu_util'].mean():.2f}%")

        # 2개의 서브플롯으로 구성된 그래프 생성
        # (2행 1열 레이아웃, 크기 12x8 인치)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 첫 번째 그래프: 메모리 사용량 추이
        ax1.plot(df["memory_used"], label="Used Memory", color="red")
        ax1.plot(df["memory_free"], label="Free Memory", color="green")
        ax1.set_xlabel("Time (seconds)")  # X축 레이블
        ax1.set_ylabel("Memory (MB)")  # Y축 레이블
        ax1.set_title("GPU Memory Usage Over Time")  # 제목
        ax1.legend()  # 범례 표시
        ax1.grid(True)  # 격자 표시

        # 두 번째 그래프: GPU 사용률 추이
        ax2.plot(df["gpu_util"], label="GPU Utilization", color="blue")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Utilization (%)")
        ax2.set_title("GPU Utilization Over Time")
        ax2.legend()
        ax2.grid(True)

        # 레이아웃 자동 조정
        plt.tight_layout()

        # 이미지 파일로 저장
        plt.savefig("gpu_analysis.png")
        print("\n그래프 저장: gpu_analysis.png")

        return df

    except Exception as e:
        print(f"로그 분석 실패: {e}")
        return None


# 메인 실행 부분
if __name__ == "__main__":
    import sys

    # 커맨드라인 인자 확인
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        # 분석 모드: 기존 로그 파일 분석
        analyze_gpu_log()
    else:
        # 모니터링 모드: 실시간 모니터링 시작
        monitor_gpu_realtime()
