#!/bin/bash
# run_full_analysis.sh
# GS-ICP-SLAM과 GLCM 텍스처 분석을 통합 실행하는 Bash 스크립트
#
# 실행 순서:
#   1. 모든 scene의 GLCM 텍스처 분석
#   2. 각 scene별로 GS-ICP-SLAM 실행 + GPU 모니터링
#   3. GPU 로그 통합 분석

echo "GS-ICP-SLAM + GLCM 텍스처 분석 파이프라인"
echo "================================================"

# Replica scene 리스트 정의
# 배열 형태로 저장
SCENES=("room0" "room1" "room2" "office0" "office1" "office2" "office3" "office4")

# 결과를 저장할 디렉토리 생성
# -p 옵션: 이미 존재하면 에러 없이 넘어감
mkdir -p /home/GS_ICP_SLAM/texture_analysis_results

# ========================================
# Step 1: GLCM 텍스처 분석 (전체 scene)
# ========================================
echo ""
echo "Step 1: GLCM 텍스처 분석"
echo "================================================"
# Python 스크립트 실행
python3 /home/GS_ICP_SLAM/analyze_replica_texture.py

# ========================================
# Step 2: 각 scene별 GS-ICP-SLAM 실행
# ========================================
# 배열의 각 요소를 순회
for scene in "${SCENES[@]}"; do
    echo ""
    echo "Processing $scene..."
    echo "================================================"
    
    # GPU 로그 파일 초기화
    # CSV 헤더 작성
    echo "timestamp,memory_used,memory_free,memory_total,gpu_util" > gpu_log_${scene}.csv
    
    # GPU 모니터링 시작 (백그라운드 실행)
    # & : 백그라운드 실행
    # >> : 파일에 추가 모드로 출력
    # $! : 마지막 백그라운드 프로세스의 PID 저장
    nvidia-smi --query-gpu=timestamp,memory.used,memory.free,memory.total,utilization.gpu \
        --format=csv,noheader,nounits -l 1 >> gpu_log_${scene}.csv &
    GPU_PID=$!
    
    # GS-ICP-SLAM 실행
    # -W ignore: Python 경고 무시
    python3 -W ignore gs_icp_slam.py \
        --dataset_path /home/dataset/Replica/${scene} \
        --output_path /home/GS_ICP_SLAM/texture_analysis_results/${scene} \
        --verbose
    
    # GPU 모니터링 프로세스 종료
    # kill 명령어로 PID에 해당하는 프로세스 중단
    kill $GPU_PID
    
    # GPU 로그 파일을 결과 디렉토리로 이동
    mv gpu_log_${scene}.csv /home/GS_ICP_SLAM/texture_analysis_results/
    
    echo "$scene 완료"
done

# ========================================
# Step 3: GPU 로그 분석
# ========================================
echo ""
echo "Step 3: GPU 메모리 분석"
echo "================================================"
# GPU 로그 통합 분석 스크립트 실행
python3 /home/GS_ICP_SLAM/analyze_gpu_logs.py

echo ""
echo "모든 분석 완료!"
echo "결과 위치: /home/GS_ICP_SLAM/texture_analysis_results/"