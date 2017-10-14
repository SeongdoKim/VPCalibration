#include "VDResidual.h"

#include <ceres\rotation.h>
#include <ceres\autodiff_cost_function.h>

VDResidual::VDResidual(double lx, double ly, double lz, double tx, double ty, int dir)
		: lx(lx), ly(ly), lz(lz), tx(tx), ty(ty), dir(dir) {
	// do nothing
}

template <typename T>
bool VDResidual::operator()(const T* const x, const T* f, T* residuals) const {
	T rot[3] = { x[0], x[1], x[2] };
	T R[9]; // (in column major order)
	ceres::AngleAxisToRotationMatrix(rot, R);

	T K[9] = { f[0], T(0), T(tx), T(0), f[0], T(ty), T(0), T(0), T(1) };
	T l[3] = { T(lx), T(ly), T(lz) };
	// u = K.t() * l -> lever vector
	T ux = K[0] * l[0] + K[3] * l[1] + K[6] * l[2];
	T uy = K[1] * l[0] + K[4] * l[1] + K[7] * l[2];
	T uz = K[2] * l[0] + K[5] * l[1] + K[8] * l[2];

	// normalize the lever vector
	T leng = sqrt(ux*ux + uy*uy + uz*uz);
	ux /= leng;
	uy /= leng;
	uz /= leng;

	residuals[0] = abs(R[3 * dir] * ux + R[3 * dir + 1] * uy + R[3 * dir + 2] * uz);

	return true;
}

ceres::CostFunction* VDResidual::Create(const double lx, const double ly, const double lz,
	const double tx, const double ty, const int dir) {
	return (new ceres::AutoDiffCostFunction<VDResidual, 1, 3, 1>(new VDResidual(lx, ly, lz, tx, ty, dir)));
}

VDResidual2::VDResidual2(double f, double lx, double ly, double lz, double tx, double ty, int dir)
	: focalLength(f), lx(lx), ly(ly), lz(lz), tx(tx), ty(ty), dir(dir) {
	// do nothing
}

template <typename T>
bool VDResidual2::operator()(const T* const x, T* residuals) const {
	T rot[3] = { x[0], x[1], x[2] };
	T R[9]; // (in column major order)
	ceres::AngleAxisToRotationMatrix(rot, R);

	T K[9] = { T(focalLength), T(0), T(tx), T(0), T(focalLength), T(ty), T(0), T(0), T(1) };
	T l[3] = { T(lx), T(ly), T(lz) };
	// u = K.t() * l -> lever vector
	T ux = K[0] * l[0] + K[3] * l[1] + K[6] * l[2];
	T uy = K[1] * l[0] + K[4] * l[1] + K[7] * l[2];
	T uz = K[2] * l[0] + K[5] * l[1] + K[8] * l[2];

	// normalize the lever vector
	T leng = sqrt(ux*ux + uy*uy + uz*uz);
	ux /= leng;
	uy /= leng;
	uz /= leng;

	residuals[0] = R[3 * dir] * ux + R[3 * dir + 1] * uy + R[3 * dir + 2] * uz;

	return true;
}

ceres::CostFunction* VDResidual2::Create(const double f, const double lx, const double ly, const double lz,
	const double tx, const double ty, const int dir) {
	return (new ceres::AutoDiffCostFunction<VDResidual2, 1, 3>(new VDResidual2(f, lx, ly, lz, tx, ty, dir)));
}
