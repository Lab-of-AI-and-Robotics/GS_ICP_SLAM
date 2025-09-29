# analyze_replica_texture_fixed.py
# 수정된 Replica 데이터셋 텍스처 분석

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
    Replica 데이터셋의 텍스처를 분석하는 클래스 (수정 버전)
    """

    def __init__(
        self, replica_base_path="/home/dataset/Replica", method="dissimilarity"
    ):
        """
        초기화 함수

        Args:
            replica_base_path: Replica 데이터셋 기본 경로
            method: 텍스처 스코어 계산 방법
        """
        self.base_path = Path(replica_base_path)
        self.method = method

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
            "contrast": 0.30,
            "dissimilarity": 0.30,
            "homogeneity": 0.20,
            "entropy": 0.20,
        }

        self.texture_weights_original = {
            "contrast": 0.25,
            "energy": 0.25,
            "homogeneity": 0.25,
            "entropy": 0.25,
        }

        # 방법별 threshold 설정 (실제 데이터 분포 기반)
        self.thresholds = {
            "dissimilarity": {"high": 8.0, "low": 5.0},      # 범위: 3.9~13.2
            "original": {"high": 7.0, "low": 4.5},           # 범위: 3.9~11.4
            "feature_selection": {"high": 0.19, "low": 0.175}, # 범위: 0.17~0.21
            "gradient": {"high": 0.7, "low": 0.5},           # 범위: 0.5~0.95
        }

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
            features["gradient_score"] = self._calculate_gradient_score(gray)

            # 선택한 방법으로 텍스처 스코어 계산
            if self.method == "dissimilarity":
                features["texture_score"] = self._calculate_texture_score_dissimilarity(
                    features
                )
            elif self.method == "original":
                features["texture_score"] = self._calculate_texture_score_original(
                    features
                )
            elif self.method == "feature_selection":
                features["texture_score"] = (
                    self._calculate_texture_score_feature_selection(features)
                )
            elif self.method == "gradient":
                features["texture_score"] = features["gradient_score"]
            else:
                features["texture_score"] = self._calculate_texture_score_dissimilarity(
                    features
                )

            # 모든 방법의 스코어 저장
            features["score_dissimilarity"] = (
                self._calculate_texture_score_dissimilarity(features)
            )
            features["score_original"] = self._calculate_texture_score_original(
                features
            )
            features["score_feature_selection"] = (
                self._calculate_texture_score_feature_selection(features)
            )
            features["score_gradient"] = features["gradient_score"]

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

    def _calculate_gradient_score(self, gray_image):
        """Gradient 기반 텍스처 스코어 계산"""
        gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx**2 + gy**2)

        hist, _ = np.histogram(gray_image, bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-8)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))

        grad_norm = np.clip(gradient_mag.mean() / 50.0, 0, 1)
        entropy_norm = entropy / 8.0

        return 0.6 * grad_norm + 0.4 * entropy_norm

    def _calculate_texture_score_dissimilarity(self, features):
        """
        Dissimilarity 기반 텍스처 스코어 (권장)
        
        모든 항이 "높을수록 텍스처 많다" 방향으로 통일
        """
        return (
            self.texture_weights["contrast"] * features["contrast"]
            + self.texture_weights["dissimilarity"] * features["dissimilarity"]
            + self.texture_weights["homogeneity"] * (1 - features["homogeneity"])
            + self.texture_weights["entropy"] * features["entropy"]
        )

    def _calculate_texture_score_original(self, features):
        """
        기존 Energy 기반 텍스처 스코어 (비교용)
        """
        return (
            self.texture_weights_original["contrast"] * features["contrast"]
            + self.texture_weights_original["energy"] * features["energy"]
            + self.texture_weights_original["homogeneity"]
            * (1 - features["homogeneity"])
            + self.texture_weights_original["entropy"] * features["entropy"]
        )

    def _calculate_texture_score_feature_selection(self, features):
        """
        특징 선택 방식 (수정됨!)
        
        수정 사항:
        - Energy를 (1-Energy)로 변경
        - 이제 모든 항이 "높을수록 텍스처 많다" 방향
        """
        return (
            0.4 * (1 - features["energy"])        # 수정!
            + 0.3 * (1 - features["correlation"])
            + 0.3 * (1 - features["homogeneity"])
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
            "gradient_score",
            "texture_score",
            "score_dissimilarity",
            "score_original",
            "score_feature_selection",
            "score_gradient",
        ]

        for key in all_keys:
            if key in features_list[0]:
                values = [f[key] for f in features_list]
                stats[f"{key}_mean"] = np.mean(values)
                stats[f"{key}_std"] = np.std(values)
                stats[f"{key}_min"] = np.min(values)
                stats[f"{key}_max"] = np.max(values)

        # 환경 분류 - 방법별로 다른 threshold 사용
        avg_score = stats["texture_score_mean"]
        thresholds = self.thresholds[self.method]

        if avg_score > thresholds["high"]:
            stats["category"] = "High-texture"
            stats["camera_weight"] = 0.8
            stats["lidar_weight"] = 0.2
        elif avg_score > thresholds["low"]:
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
        print(f"Replica 데이터셋 텍스처 분석 (방법: {self.method})")
        print("=" * 60)

        all_stats = []

        for scene in self.scenes:
            stats = self.analyze_scene(scene, sample_count=50)
            if stats:
                all_stats.append(stats)
                self._print_scene_stats(stats)

        output_file = f"replica_texture_analysis_{self.method}_fixed.json"
        with open(output_file, "w") as f:
            json.dump(all_stats, f, indent=2)

        print(f"\n결과 저장: {output_file}")

        self._visualize_comparison(all_stats)

        if self.method == "dissimilarity":
            self._visualize_method_comparison(all_stats)

        return all_stats

    def _print_scene_stats(self, stats):
        """단일 scene의 통계를 출력"""
        print(f"\n{'='*50}")
        print(f"{stats['scene']}")
        print(f"{'='*50}")
        print(
            f"  텍스처 스코어 ({self.method}): {stats['texture_score_mean']:.3f} (±{stats['texture_score_std']:.3f})"
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
            f"    Dissimilarity:  {stats['dissimilarity_mean']:.3f} (±{stats['dissimilarity_std']:.3f})"
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

        if "score_dissimilarity_mean" in stats:
            print(f"\n  방법별 스코어 비교:")
            print(f"    Dissimilarity 방식: {stats['score_dissimilarity_mean']:.3f}")
            print(f"    Original 방식:      {stats['score_original_mean']:.3f}")
            print(
                f"    Feature Selection:  {stats['score_feature_selection_mean']:.3f}"
            )
            print(f"    Gradient 방식:      {stats['score_gradient_mean']:.3f}")

    def _visualize_comparison(self, all_stats):
        """모든 scene의 텍스처 특징을 비교하는 시각화"""
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
        
        # 방법별 threshold 표시
        thresholds = self.thresholds[self.method]
        ax1.axhline(
            y=thresholds["high"],
            color="green",
            linestyle="--",
            label="High threshold",
        )
        ax1.axhline(
            y=thresholds["low"], color="red", linestyle="--", label="Low threshold"
        )
        
        ax1.set_xlabel("Scene", fontsize=12)
        ax1.set_ylabel("Texture Score", fontsize=12)
        ax1.set_title(
            f"Texture Score Comparison ({self.method})", fontsize=14, fontweight="bold"
        )
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 그래프 2: Contrast vs Dissimilarity
        ax2 = axes[0, 1]
        dissimilarities = [s["dissimilarity_mean"] for s in all_stats]
        contrasts = [s["contrast_mean"] for s in all_stats]

        x = np.arange(len(scenes))
        width = 0.35
        ax2.bar(
            x - width / 2,
            contrasts,
            width,
            label="Contrast",
            color="skyblue",
            alpha=0.7,
        )
        ax2.bar(
            x + width / 2,
            dissimilarities,
            width,
            label="Dissimilarity",
            color="coral",
            alpha=0.7,
        )
        ax2.set_xlabel("Scene", fontsize=12)
        ax2.set_ylabel("Value", fontsize=12)
        ax2.set_title("Contrast vs Dissimilarity", fontsize=14, fontweight="bold")
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
        output_file = f"replica_texture_comparison_{self.method}_fixed.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"비교 그래프 저장: {output_file}")
        plt.close()

    def _visualize_method_comparison(self, all_stats):
        """여러 텍스처 스코어 계산 방법을 비교"""
        if not all_stats:
            return

        scenes = [s["scene"] for s in all_stats]

        score_dissimilarity = [s["score_dissimilarity_mean"] for s in all_stats]
        score_original = [s["score_original_mean"] for s in all_stats]
        score_feature_sel = [s["score_feature_selection_mean"] for s in all_stats]
        score_gradient = [s["score_gradient_mean"] for s in all_stats]

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(scenes))
        width = 0.2

        ax.bar(
            x - 1.5 * width,
            score_dissimilarity,
            width,
            label="Dissimilarity (권장)",
            color="green",
            alpha=0.7,
        )
        ax.bar(
            x - 0.5 * width,
            score_original,
            width,
            label="Original (Energy)",
            color="red",
            alpha=0.7,
        )
        ax.bar(
            x + 0.5 * width,
            score_feature_sel,
            width,
            label="Feature Selection (수정)",
            color="blue",
            alpha=0.7,
        )
        ax.bar(
            x + 1.5 * width,
            score_gradient,
            width,
            label="Gradient",
            color="orange",
            alpha=0.7,
        )

        ax.set_title("Texture Score Method Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(scenes, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig("replica_method_comparison_fixed.png", dpi=300, bbox_inches="tight")
        print(f"방법 비교 그래프 저장: replica_method_comparison_fixed.png")
        plt.close()


def compare_all_methods():
    """모든 텍스처 계산 방법을 실행하고 비교"""
    methods = ["dissimilarity", "original", "feature_selection", "gradient"]
    all_results = {}

    for method in methods:
        print(f"\n\n{'='*70}")
        print(f"방법: {method} 실행 중...")
        print(f"{'='*70}")

        analyzer = ReplicaTextureAnalyzer(method=method)
        results = analyzer.analyze_all_scenes()
        all_results[method] = results

    with open("replica_all_methods_comparison_fixed.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\n모든 방법 비교 완료!")
    print("결과: replica_all_methods_comparison_fixed.json")


if __name__ == "__main__":
    # 기본 실행: Dissimilarity 방법 (권장)
    analyzer = ReplicaTextureAnalyzer(method="original")
    results = analyzer.analyze_all_scenes()

    # 모든 방법 비교
    #compare_all_methods()