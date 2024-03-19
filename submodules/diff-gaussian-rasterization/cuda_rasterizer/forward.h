/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

struct cov6{
	float xx; float xy; float zx;
		 float yy; float yz;
		           float zz;
};

struct float6{
	float x; float y; float z; float w;
	float i; float j;
};


namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths, // added
		float* cov3Ds,
		float* colors,
		float6* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		bool* is_used); //added

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* depths, // added
		const float* features,
		const float6* conic_opacity,
		float* final_T,
		float* final_T_d,
		float* final_accum_prob_d,
		uint32_t* n_contrib,
		float* final_depth,
		const float* bg_color,
		float* out_depth, // added 
		float* out_color);
}


#endif
