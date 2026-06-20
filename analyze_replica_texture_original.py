# analyze_replica_texture_original.py
# Replica 데이터셋 텍스처 분석 (Original 방법만)

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class ReplicaTextureAnalyzer:
    """
    Replica 데이터셋의 텍스처를 분석하는 클래스 (Original 방법)
    """

    def __init__(self, replica_base_path="/home/dataset/Replica"):
        """
        초기화 함수

        Args:
            replica_base_path: Replica 데이터셋 기본 경로
        """
        self.base_path = Path(replica_base_path)

        # 분석할 scene 목록
        self.scenes = [
            "room0",
            "room1",
            "room2",
            "office0",
            "office1",
            "office2",
            "office3",
            "office4",
        ]

        # 텍스처 특징 가중치
        self.texture_weights = {
            "contrast": 0.25,
            "energy": 0.25,
            "homogeneity": 0.25,
            "entropy": 0.25,
        }

        # Threshold 설정
        self.threshold_high = 7.0
        self.threshold_low = 4.5

    def analyze_scene(self, scene_name, sample_count=50):
        """특정 scene의 텍스처를 분석"""
        print(f"\n{scene_name} 분석 중...")

        scene_path = self.base_path / scene_name / "images"

        if not scene_path.exists():
            print(f"경로를 찾을 수 없음: {scene_path}")
            return None

        image_files = sorted(list(scene_path.glob("*.jpg")))

        if not image_files:
            print(f"이미지를 찾을 수 없음: {scene_path}")
            return None

        step = max(1, len(image_files) // sample_count)
        sampled_files = image_files[::step][:sample_count]

        print(f"  전체 이미지 수: {len(image_files)}")
        print(f"  분석할 샘플 수: {len(sampled_files)}")

        all_features = []

        for img_file in sampled_files:
            features = self._analyze_single_image(img_file)
            if features:
                all_features.append(features)

        if not all_features:
            print(f"이미지 분석 실패")
            return None

        stats = self._calculate_statistics(all_features, scene_name)

        return stats

    def _analyze_single_image(self, image_path):
        """단일 이미지에서 텍스처 특징을 추출"""
        try:
            image = cv2.imread(str(image_path))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (256, 256))
            gray_quantized = (gray // 4).astype(np.uint8)

            glcm = graycomatrix(
                gray_quantized,
                distances=[1, 2, 3],
                angles=np.radians([0, 45, 90, 135]),
                levels=64,
                symmetric=True,
                normed=True,
            )

            features = {
                "contrast": graycoprops(glcm, "contrast").mean(),
                "dissimilarity": graycoprops(glcm, "dissimilarity").mean(),
                "energy": graycoprops(glcm, "energy").mean(),
                "homogeneity": graycoprops(glcm, "homogeneity").mean(),
                "correlation": graycoprops(glcm, "correlation").mean(),
            }

            features["entropy"] = self._calculate_entropy(glcm)
            features["texture_score"] = self._calculate_texture_score(features)

            return features

        except Exception as e:
            print(f"  오류 - {image_path.name}: {e}")
            return None

    def _calculate_entropy(self, glcm):
        """GLCM에서 엔트로피를 계산"""
        glcm_norm = glcm / (np.sum(glcm) + 1e-8)
        glcm_nonzero = glcm_norm[glcm_norm > 0]

        if len(glcm_nonzero) > 0:
            return -np.sum(glcm_nonzero * np.log2(glcm_nonzero))
        return 0

    def _calculate_texture_score(self, features):
        """
        Original Energy 기반 텍스처 스코어
        """
        return (
            self.texture_weights["contrast"] * features["contrast"]
            + self.texture_weights["energy"] * features["energy"]
            + self.texture_weights["homogeneity"] * (1 - features["homogeneity"])
            + self.texture_weights["entropy"] * features["entropy"]
        )

    def _calculate_statistics(self, features_list, scene_name):
        """여러 이미지의 특징들로부터 통계를 계산"""
        stats = {"scene": scene_name}

        all_keys = [
            "contrast",
            "dissimilarity",
            "energy",
            "homogeneity",
            "correlation",
            "entropy",
            "texture_score",
        ]

        for key in all_keys:
            if key in features_list[0]:
                values = [f[key] for f in features_list]
                stats[f"{key}_mean"] = np.mean(values)
                stats[f"{key}_std"] = np.std(values)
                stats[f"{key}_min"] = np.min(values)
                stats[f"{key}_max"] = np.max(values)

        # 환경 분류
        avg_score = stats["texture_score_mean"]

        if avg_score > self.threshold_high:
            stats["category"] = "High-texture"
            stats["camera_weight"] = 0.8
            stats["lidar_weight"] = 0.2
        elif avg_score > self.threshold_low:
            stats["category"] = "Medium-texture"
            stats["camera_weight"] = 0.5
            stats["lidar_weight"] = 0.5
        else:
            stats["category"] = "Low-texture"
            stats["camera_weight"] = 0.2
            stats["lidar_weight"] = 0.8

        return stats

    def analyze_all_scenes(self):
        """모든 scene을 분석하는 메인 함수"""
        print("=" * 60)
        print("Replica 데이터셋 텍스처 분석 (Original 방법)")
        print("=" * 60)

        all_stats = []

        for scene in self.scenes:
            stats = self.analyze_scene(scene, sample_count=50)
            if stats:
                all_stats.append(stats)
                self._print_scene_stats(stats)

        output_file = "replica_texture_analysis_original.json"
        with open(output_file, "w") as f:
            json.dump(all_stats, f, indent=2)

        print(f"\n결과 저장: {output_file}")

        self._visualize_results(all_stats)

        return all_stats

    def _print_scene_stats(self, stats):
        """단일 scene의 통계를 출력"""
        print(f"\n{'='*50}")
        print(f"{stats['scene']}")
        print(f"{'='*50}")
        print(
            f"  텍스처 스코어: {stats['texture_score_mean']:.3f} (±{stats['texture_score_std']:.3f})"
        )
        print(f"  환경 분류: {stats['category']}")
        print(
            f"  센서 가중치: Camera {stats['camera_weight']}, LiDAR {stats['lidar_weight']}"
        )
        print(f"\n  텍스처 특징:")
        print(
            f"    Contrast:       {stats['contrast_mean']:.3f} (±{stats['contrast_std']:.3f})"
        )
        print(
            f"    Energy:         {stats['energy_mean']:.3f} (±{stats['energy_std']:.3f})"
        )
        print(
            f"    Homogeneity:    {stats['homogeneity_mean']:.3f} (±{stats['homogeneity_std']:.3f})"
        )
        print(
            f"    Entropy:        {stats['entropy_mean']:.3f} (±{stats['entropy_std']:.3f})"
        )

    def _visualize_results(self, all_stats):
        """분석 결과를 시각화"""
        if not all_stats:
            return

        scenes = [s["scene"] for s in all_stats]
        texture_scores = [s["texture_score_mean"] for s in all_stats]
        categories = [s["category"] for s in all_stats]

        color_map = {
            "High-texture": "green",
            "Medium-texture": "orange",
            "Low-texture": "red",
        }
        colors = [color_map[cat] for cat in categories]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 그래프 1: 텍스처 스코어 비교
        ax1 = axes[0, 0]
        ax1.bar(scenes, texture_scores, color=colors, alpha=0.7, edgecolor="black")
        ax1.axhline(
            y=self.threshold_high,
            color="green",
            linestyle="--",
            label="High threshold",
        )
        ax1.axhline(
            y=self.threshold_low, color="red", linestyle="--", label="Low threshold"
        )
        ax1.set_xlabel("Scene", fontsize=12)
        ax1.set_ylabel("Texture Score", fontsize=12)
        ax1.set_title("Texture Score Comparison", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 그래프 2: 텍스처 특징 비교
        ax2 = axes[0, 1]
        contrasts = [s["contrast_mean"] for s in all_stats]
        energies = [s["energy_mean"] for s in all_stats]

        x = np.arange(len(scenes))
        width = 0.35
        ax2.bar(
            x - width / 2, contrasts, width, label="Contrast", color="skyblue", alpha=0.7
        )
        ax2.bar(
            x + width / 2, energies, width, label="Energy", color="coral", alpha=0.7
        )
        ax2.set_xlabel("Scene", fontsize=12)
        ax2.set_ylabel("Value", fontsize=12)
        ax2.set_title("Contrast vs Energy", fontsize=14, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenes, rotation=45, ha="right")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

        # 그래프 3: 센서 가중치 분포
        ax3 = axes[1, 0]
        camera_weights = [s["camera_weight"] for s in all_stats]
        lidar_weights = [s["lidar_weight"] for s in all_stats]

        x = np.arange(len(scenes))
        width = 0.35
        ax3.bar(
            x - width / 2,
            camera_weights,
            width,
            label="Camera",
            color="blue",
            alpha=0.7,
        )
        ax3.bar(
            x + width / 2, lidar_weights, width, label="LiDAR", color="red", alpha=0.7
        )
        ax3.set_xlabel("Scene", fontsize=12)
        ax3.set_ylabel("Weight", fontsize=12)
        ax3.set_title("Sensor Weight Distribution", fontsize=14, fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(scenes, rotation=45, ha="right")
        ax3.legend()
        ax3.grid(axis="y", alpha=0.3)

        # 그래프 4: 환경 분류 분포
        ax4 = axes[1, 1]
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1

        ax4.pie(
            category_counts.values(),
            labels=category_counts.keys(),
            autopct="%1.1f%%",
            colors=[color_map[cat] for cat in category_counts.keys()],
            startangle=90,
        )
        ax4.set_title(
            "Environment Category Distribution", fontsize=14, fontweight="bold"
        )

        plt.tight_layout()
        output_file = "replica_texture_analysis_original.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"분석 그래프 저장: {output_file}")
        plt.close()


if __name__ == "__main__":
    analyzer = ReplicaTextureAnalyzer()
    results = analyzer.analyze_all_scenes()