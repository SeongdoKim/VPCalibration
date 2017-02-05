#include <stdio.h>
#include <math.h>
#include <vector>

#include <cublas_v2.h>

#include <thrust\device_vector.h>
#include <thrust\device_ptr.h>
#include <thrust\fill.h>
#include <thrust\reduce.h>
#include <thrust\extrema.h>

#include "CUDAHelper.h"
#include "TLinkageGPU.h"

#define MAX_THREADS_NUM	768


inline __device__
void vec_cross(float a1, float b1, float c1,
	float a2, float b2, float c2,
	float& a3, float& b3, float& c3) {
	a3 = b1*c2 - c1*b2;
	b3 = -(a1*c2 - c1*a2);
	c3 = a1*b2 - b1*a2;
}

inline __device__
float LineDistance(const float* line, const float* model) {
	float l[3];
	float midPoint[3] = { (line[0] + line[2]) / 2.0, (line[1] + line[3]) / 2.0, 1 };

	vec_cross(midPoint[0], midPoint[1], midPoint[2],
		model[0], model[1], model[2],
		l[0], l[1], l[2]);

	return fabs(l[0] * line[0] + l[1] * line[1] + l[2]) / sqrt(l[0] * l[0] + l[1] * l[1]);
}

inline __device__
float computeTonimotoDistance(float *vec1, float *vec2, int dim) {
	float s = 0.f, n1 = 0.f, n2 = 0.f;

	for (int i = 0; i < dim; i++) {
		s += vec1[i] * vec2[i];
		n1 += vec1[i] * vec1[i];
		n2 += vec2[i] * vec2[i];
	}

	float partial_sum = n1 + n2 - s;

	if (partial_sum <= 0.f) {
		return 1.f;
	}
	else {
		return 1.f - s / partial_sum;
	}
}

__global__
void computePrefMat(float *prefMat, float* line, float* model, float epsilon, int cols) {
	int modelIdx = blockIdx.x;		// column index
	int lineIdx = threadIdx.x;		// row index

	float residual = LineDistance(&line[lineIdx * 4], &model[modelIdx * 9]);
	if (residual < epsilon) {
		float tau = epsilon / 5.f;
		prefMat[lineIdx * cols + modelIdx] = exp(-residual / tau);
	}
}

__global__
void computeTonimotoDistance(float *prefMat, int dim, int *U, int *V, float *distances, int N) {
	int index = blockIdx.x * MAX_THREADS_NUM + threadIdx.x;

	if (index >= N) {
		// If current index is larger than N, then this is a void thread
		return;
	}

	distances[index] = computeTonimotoDistance(&prefMat[U[index] * dim], &prefMat[V[index] * dim], dim);
}

__global__
void computeIndices(int *U, int *V, int m, int N) {
	int idx = blockIdx.x * MAX_THREADS_NUM + threadIdx.x;

	if (idx >= N) {
		// If current index is larger than N, then this is a void thread
		return;
	}

	U[idx] = m - (int)round(sqrt(float(2 * (N - idx)))) - 1;
	V[idx] = (idx + (U[idx] + 1) * (U[idx] + 2) / 2) % m;
}

__global__
void updateDistance(float *vec_tar, float *vec1, float *vec2, int dim) {
	int i = blockIdx.x * MAX_THREADS_NUM + threadIdx.x;

	if (i < dim) {
		vec_tar[i] = vec1[i] < vec2[i] ? vec1[i] : vec2[i];
	}
}

__global__
void update(int *U, int *V, int x, int y, int *flags, float *prefMat,
float *distances, int N, int n, int k, int dim) {
	int idx = blockIdx.x * MAX_THREADS_NUM + threadIdx.x;	// [0, N)

	if (idx >= N) {
		// If current index is larger than N, then this is a void thread
		return;
	}

	if (U[idx] == y || V[idx] == y) {
		flags[idx] = 0;
	}

	if (flags[idx]) {
		bool update = false;

		if (U[idx] == x) {
			U[idx] = n + k;
			update = true;
		}
		else if (V[idx] == x) {
			V[idx] = n + k;
			update = true;
		}

		// update distances as well
		if (update) {
			distances[idx] = computeTonimotoDistance(&prefMat[U[idx] * dim], &prefMat[V[idx] * dim], dim);
		}
	}

	// instead of make submatrix of distances, make their values to be larger than 1.0.
	if (flags[idx] == 0 && distances[idx] < 1.0f) {
		distances[idx] = 2.0f;
	}
}

extern "C" void TLinkageGPU(float *lines, int numLines,
	float *models, int numModels, float epsilon, float *prefMat,
	std::vector<std::pair<int, int>> &T) {

	// Variables
	int k = 0;	// iteration
	const int N = (numLines * (numLines - 1) / 2);
	int m = ceil(sqrt(float(2 * N)));
	int numOfBlocks = (N - 1) / MAX_THREADS_NUM + 1;	// #blocks for computing linear memory
	cudaEvent_t start, stop;	// measuring time
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Memory sizes
	size_t lengthLines = sizeof(float) * numLines * 4;
	size_t lengthModels = sizeof(float) * numModels * 3;
	size_t lengthPrefMat = sizeof(float) * numLines * numModels * 2;

	// GPU memory pointers
	float *gpuLines;		// Array of lines on GPU side
	float *gpuModels;		// Array of models
	float *gpuPrefMat;		// Preference matrix

	printf("T-linkage clustering on GPU...\n");

	// GPU memory via thrust vector
	thrust::device_vector<int> flags(N, 1);
	thrust::device_vector<int> U(N);
	thrust::device_vector<int> V(N);
	thrust::device_vector<float> distances(N);

	cudaEventRecord(start, 0);

	printf("Manually allocating GPU memories and copying input data...\n");
	cudaErrorCheck(cudaMalloc((void **)&gpuLines, lengthLines));
	cudaErrorCheck(cudaMalloc((void **)&gpuModels, lengthModels));
	cudaErrorCheck(cudaMalloc((void **)&gpuPrefMat, lengthPrefMat));

	// Initialize memory
	cudaErrorCheck(cudaMemset(gpuPrefMat, 0, lengthPrefMat));

	// Copy CPU data to GPU memory
	cudaErrorCheck(cudaMemcpy(gpuLines, lines, lengthLines, cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpy(gpuModels, models, lengthModels, cudaMemcpyHostToDevice));

	// Compute preference matrix
	computePrefMat<<<numModels, numLines>>>(gpuPrefMat, gpuLines, gpuModels, epsilon, numModels);

	// Compute indices
	computeIndices<<<numOfBlocks, MAX_THREADS_NUM>>>(thrust::raw_pointer_cast(&U[0]),
		thrust::raw_pointer_cast(&V[0]), m, N);

	// Compute pairwise Tonimoto distance
	computeTonimotoDistance<<<numOfBlocks, MAX_THREADS_NUM>>>(gpuPrefMat, numModels,
		thrust::raw_pointer_cast(&U[0]), thrust::raw_pointer_cast(&V[0]),
		thrust::raw_pointer_cast(&distances[0]), N);
	
	while (thrust::reduce(flags.begin(), flags.end())) {
		// Find the minimum position
		thrust::device_vector<float>::iterator iter = thrust::min_element(distances.begin(), distances.end());
		unsigned int minIndex = iter - distances.begin();
		if (distances[minIndex] >= 1.0f) {
			break;
		}

		// Get indices of points for the found position
		int x = U[minIndex];
		int y = V[minIndex];

		T.push_back({ x, y });

		// Compute new preference set (ps) after merging two pss.
		updateDistance<<<(numModels - 1) / MAX_THREADS_NUM + 1, MAX_THREADS_NUM>>>(
			&gpuPrefMat[(numLines + k) * numModels],
			&gpuPrefMat[x * numModels], &gpuPrefMat[y * numModels], numModels);
		
		// Update matrices
		update<<<numOfBlocks, MAX_THREADS_NUM>>>(thrust::raw_pointer_cast(&U[0]),
			thrust::raw_pointer_cast(&V[0]),
			x, y, thrust::raw_pointer_cast(&flags[0]), gpuPrefMat,
			thrust::raw_pointer_cast(&distances[0]), N, numLines, k, numModels);

		k++;

		if (k >= numLines) {
			break;
		}
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	printf("Total number of iterations: %d\n", k);
	printf("GPU computing takes %f ms\n", time);

	// Copy result to CPU memory
	cudaErrorCheck(cudaMemcpy(prefMat, gpuPrefMat, lengthPrefMat, cudaMemcpyDeviceToHost));

	// Release GPU memories
	cudaErrorCheck(cudaFree(gpuLines));
	cudaErrorCheck(cudaFree(gpuModels));
	cudaErrorCheck(cudaFree(gpuPrefMat));
}
