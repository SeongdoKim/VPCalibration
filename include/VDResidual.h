#pragma once

#include <ceres\cost_function.h>

/**
 * Residual class to refine vanishing directions
 */
struct VDResidual {
	VDResidual(double lx, double ly, double lz, double tx, double ty, int dir);

	template <typename T>
	bool operator()(const T* const x, const T* f, T* residuals) const;

	static ceres::CostFunction* Create(const double lx, const double ly, const double lz,
		const double tx, const double ty, const int dir);

	double lx, ly, lz; // normal vector of a line
	double tx, ty;     // image center
	int dir;           // direction of line (0, 1, 2) == (x, y, z) direction
};

/**
 * Residual class to refine camera focal length and vanishing directions
 */
struct VDResidual2 {
	VDResidual2(double f, double lx, double ly, double lz, double tx, double ty, int dir);

	template <typename T>
	bool operator()(const T* const x, T* residuals) const;

	static ceres::CostFunction* Create(const double f, const double lx, const double ly, const double lz,
		const double tx, const double ty, const int dir);

	double lx, ly, lz;	// normal vector of a line
	double tx, ty;		// image center
	int dir;			// direction of line (0, 1, 2) == (x, y, z) direction
	double focalLength;	// focal length
};
