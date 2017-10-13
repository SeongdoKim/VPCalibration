#pragma once

#include <vector>

extern "C" void TLinkageGPU(float *lines, int numLines,
	float *models, int numModels, float epsilon, float *prefMat,
	std::vector<std::pair<int, int>> &T);
